
# pytorch references ...
# and pytorch multi-gpu support

import torch;
from torch.autograd import Variable;
import torch.nn.functional as F;
import torch.nn;
import torch.optim as optim
from torch.optim import sgd
from torch.optim.sgd import SGD;
import torch.distributions as dist
import torch.cuda
from torch.utils.data import DataLoader, Dataset
import torch.utils.data.distributed as utilsDataDist

# image processing ...

from PIL import Image;
import numpy
import torchvision.transforms as transforms;
import os;
import matplotlib.pyplot as plt

# python utilities ...

import random
import zipfile
import argparse
import pprint
from distutils.log import error, warn
from math import sqrt
from enum import Enum
from models.backbones.convolutions import Convolution
from models.backbones.residuals import BNMOMENTUM, ResNet, ResNetSpec, ResNetTerminal, Residual

from models.backbones.utility import clampSigmoid, convolutionConv1x1, extractTopK, nonMaximumSuppression, reshapeGatherFeatures
from models.losses.focal import focalLoss
from models.losses.regression import smoothL1LossMask
from models.backbones.stackHourglass import HourglassTerminal, StackHourglass
from evaluations.detection import averageIoU, averagePrecision
from configuration import defaultConfig
from datasets.scds.scdx16p100 import DOWNSAMPLE

CLASSDIMENSION = 1
SIZEREGRFACTOR = 10

# a simplified CenterNet for object detection consists of one center gaussian heatmap
# branch and a regression branch for object size. this implementation do not use the 
# offset regression branch, since the samples is used at its original resolution.

makeHeatmapLayerHg = convolutionConv1x1

def heatmapInitializerHg(layer):
    layer[-1].bias.data.fill_(-2.19)

def process(inp, module, *xs, **kwargs):
    return module(inp)

heatmapTerminalHg = HourglassTerminal("heatmap", CLASSDIMENSION, heatmapInitializerHg, makeHeatmapLayerHg, process)

makeObjectSizeRegressionLayerHg = convolutionConv1x1

sizeRegressionTerminalHg = HourglassTerminal("size", 2, None, makeObjectSizeRegressionLayerHg, process)

def makePoolLayer(dim):
    return torch.nn.Sequential()

def makeHourglassLayer(kernelSize, inputDimension, outputDimension, modules, layer = Convolution, **kwargs):
    layers  = [layer(kernelSize, inputDimension, outputDimension, stride = 2)]
    layers += [layer(kernelSize, outputDimension, outputDimension) for _ in range(modules - 1)]
    return torch.nn.Sequential(*layers)

class CenterNetHourglass(StackHourglass):
    def __init__(self, **kwargs):

        hourglassIters     = 5
        dimensions         = [256, 256, 384, 384, 384, 512]
        modules            = [2, 2, 2, 2, 2, 4]
        outputDimensions   = CLASSDIMENSION

        super(CenterNetHourglass, self).__init__(
            hourglassIters, 1, dimensions, modules, outputDimensions,
            hourglassPool = makePoolLayer,
            hourglassBefore = makeHourglassLayer,
            hourglassLayer = Residual, predictionConvDim = 256,

            beforeBackbone = torch.nn.Sequential(
                Convolution ( 7, 1, 128 ),
                # Residual ( 3, 128, 256, stride = 2 )
                Residual ( 3, 128, dimensions[0], stride = 2 ) # CHANGE
            ),

            terminals = [heatmapTerminalHg, sizeRegressionTerminalHg],
            decoder = decodeCenterNet
        )

def makeResnetTerminal(prediction, current, output):

    if current > 0:
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(prediction, current, kernel_size = 3, padding = 1, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(current, output, kernel_size = 1, 
                            stride = 1, padding = 0))
    else:
        layer = torch.nn.Conv2d (
                    in_channels     = prediction,
                    out_channels    = output,
                    kernel_size     = 1,
                    stride          = 1,
                    padding         = 0
                )
    
    if torch.cuda.is_available() and defaultConfig.useGPU:
        return layer.cuda()
    return layer

def heatmapInitializerRes(m):
    torch.nn.init.constant_(m.bias, -2.19)

def regressionInitializerRes(m):
    torch.nn.init.normal_(m.weight, std = 0.001)
    torch.nn.init.constant_(m.bias, 0)

def makeResidualPreprocess(inputDimension):
    return torch.nn.Sequential(
               torch.nn.Conv2d(inputDimension, 64, kernel_size = 7, stride = 1, padding = 3, bias = False),
               torch.nn.BatchNorm2d(64, momentum = BNMOMENTUM),
               torch.nn.ReLU(inplace = True)
           )

resnetHeatmapTerminal = ResNetTerminal("heatmap", CLASSDIMENSION, 128, heatmapInitializerRes, makeResnetTerminal, process)
resnetSizeTerminal = ResNetTerminal("size", 2, 128, regressionInitializerRes, makeResnetTerminal, process)

class CenterNetResidual(ResNet):

    def __init__(self, numLayers):
        
        blockType, layers = ResNetSpec[numLayers]
        inputDimension = 1
        super(CenterNetResidual, self).__init__(inputDimension, blockType, layers, 

            # by following the default set of preprocess, we can downsample the resolution
            # from 512x to 128x, decreasing the GPU memory requirement per image.
            # preprocess = makeResidualPreprocess,
            
            terminals = [resnetHeatmapTerminal, resnetSizeTerminal],
            decoder = decodeCenterNet )

        self.initialize(numLayers)

class CenterNetLoss(torch.nn.Module):

    def __init__( self, regressionWeight = 1, 
        focal = focalLoss):

        super(CenterNetLoss, self).__init__()

        self.regressionWeight = regressionWeight
        self.focal = focal
        self.mse = torch.nn.MSELoss()
        self.regression = smoothL1LossMask

    def forward(self, outs, targets):
        heats = []
        regressions = []
        for out in outs:
            heats += [out["heatmap"]]
            regressions += [reshapeGatherFeatures(out["size"], targets[3])]

        groundTruthHeatmap = targets[0]
        groundTruthMask = targets[1]
        groundTruthRegression = targets[2]

        # focal loss of heatmap
        focalL = 0

        heats = [clampSigmoid(t) for t in heats]

        focalL += self.focal(heats, groundTruthHeatmap)

        offsetL = 0
        for regr in regressions:
            offsetL += self.regression(regr, groundTruthRegression / (DOWNSAMPLE * SIZEREGRFACTOR), groundTruthMask)
        offsetL = self.regressionWeight * offsetL

        loss = (focalL + offsetL) / len(heats)

        # the loss is one real number, this give a dimension to be a one-element array.
        # torch.tensor([loss]).
        return loss.unsqueeze(0), [focalL, offsetL]

def decodeCenterNet( outputDictionary, K = 100, nmsKernelSize = 3 ):

    heatmap = outputDictionary["heatmap"]
    regression = outputDictionary["size"]

    batch, category, height, width = heatmap.size()

    # apply sigmoid uniform to the heatmap, limiting its value to [0, 1].
    heatmap = torch.sigmoid(heatmap)

    # perform Non-Maximum Suppression (NMS) on heatmaps
    heatmap = nonMaximumSuppression(heatmap, kernelSize = nmsKernelSize)

    # all of these are in format of (batch, k)
    ctScores, ctIndices, ctCategories, ctY, ctX = extractTopK(heatmap, K = K)

    if regression is not None:
        
        # after the reshape feature gathering, the regression now have (batch, k, dim = 2)
        regression = reshapeGatherFeatures(regression, ctIndices)

    # the output dimensionality are listed as follows:
    # 'ctScores'   : (BATCH, K)
    # 'ctIndices'  : (BATCH, K)
    # 'ctX', 'ctY' : (BATCH, K)
    # 'regression' : (BATCH, K, 2).
    # outputs the objects with top-K scores.

    return [ctScores, ctIndices.long(), ctY.long(), ctX.long(), regression, outputDictionary]

def centerNetEvaluation(xs, ys, ctScores, ctIndices, ctY, ctX, regression, outputDictionary):
    BATCH, K = ctX.shape
    _, MAXTAGLEN, _ = ys[3].shape
    groundTruthHeat = ys[0]
    objNum = [torch.sum(mask).item() for mask in ys[1]]

    # generates the bounding boxes
    bounds = torch.zeros(BATCH, K, 4)
    bounds[:,:,0] = ctX - 0.5 * regression[:,:,0] * SIZEREGRFACTOR
    bounds[:,:,1] = ctY - 0.5 * regression[:,:,1] * SIZEREGRFACTOR
    bounds[:,:,2] = ctX + 0.5 * regression[:,:,0] * SIZEREGRFACTOR
    bounds[:,:,3] = ctY + 0.5 * regression[:,:,1] * SIZEREGRFACTOR

    # we only selects the predicted boundboxes with a score higher than 0.5 possibility.
    validMask = ctScores >= 0.5

    # calculate the ground-truth from (center-x, center-y, w, h) to (tl-x, tl-y, br-x, br-y).
    groundTruthLocs = torch.zeros(BATCH, MAXTAGLEN, 4)
    groundTruthLocs[:,:,0] = (ys[3][:,:,0] - 0.5 * ys[3][:,:,2]) / DOWNSAMPLE
    groundTruthLocs[:,:,1] = (ys[3][:,:,1] - 0.5 * ys[3][:,:,3]) / DOWNSAMPLE 
    groundTruthLocs[:,:,2] = (ys[3][:,:,0] + 0.5 * ys[3][:,:,2]) / DOWNSAMPLE 
    groundTruthLocs[:,:,3] = (ys[3][:,:,1] + 0.5 * ys[3][:,:,3]) / DOWNSAMPLE 
    if torch.cuda.is_available() and defaultConfig.useGPU:
        bounds = bounds.cuda()
        groundTruthLocs = groundTruthLocs.cuda()

    return { 'mIoU': averageIoU(bounds, groundTruthLocs, validMask),
             'ap30': averagePrecision(ctX, ctY, groundTruthHeat, objNum, 0.3),
             'ap50': averagePrecision(ctX, ctY, groundTruthHeat, objNum, 0.5),
             'ap75': averagePrecision(ctX, ctY, groundTruthHeat, objNum, 0.75),
             'ap90': averagePrecision(ctX, ctY, groundTruthHeat, objNum, 0.9)}, outputDictionary