
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
from models.losses.regression import smoothL1LossMask, L1LossMask
from models.backbones.stackHourglass import HourglassTerminal, StackHourglass
from evaluations.detection import MAE, IoU, IoUConfidence, Orthogonity
from configuration import defaultConfig
from datasets.scds.scdx16p100 import DOWNSAMPLE, HEATMAPSIZE

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

sizeRegressionTerminalHg = HourglassTerminal("regr", 4, None, makeObjectSizeRegressionLayerHg, process)

makeOffsetRegressionLayerHg = convolutionConv1x1

offsetRegressionTerminalHg = HourglassTerminal("offset", 2, None, makeOffsetRegressionLayerHg, process)

def makePoolLayer(dim):
    return torch.nn.Sequential()

def makeHourglassLayer(kernelSize, inputDimension, outputDimension, modules, layer = Convolution, **kwargs):
    layers  = [layer(kernelSize, inputDimension, outputDimension, stride = 2)]
    layers += [layer(kernelSize, outputDimension, outputDimension) for _ in range(modules - 1)]
    return torch.nn.Sequential(*layers)

class CenterNetHourglass(StackHourglass):
    def __init__(self, **kwargs):

        hourglassIters     = 5
        # dimensions         = [256, 256, 384, 384, 384, 512] # original
        dimensions         = [128, 128, 192, 192, 192, 256]
        modules            = [2, 2, 2, 2, 2, 4]
        outputDimensions   = CLASSDIMENSION

        super(CenterNetHourglass, self).__init__(
            hourglassIters, 1, dimensions, modules, outputDimensions,
            hourglassPool = makePoolLayer,
            hourglassBefore = makeHourglassLayer,
            hourglassLayer = Residual, predictionConvDim = 256,

            beforeBackbone = torch.nn.Sequential(
                Convolution ( 7, 1, 128, stride = 2 ),
                # Residual ( 3, 128, 256, stride = 2 )
                Residual ( 3, 128, dimensions[0], stride = 2 ) # CHANGE
            ),

            terminals = [heatmapTerminalHg, sizeRegressionTerminalHg, offsetRegressionTerminalHg],
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

def makeResidualPreprocess1x(inputDimension):
    return torch.nn.Sequential(
               torch.nn.Conv2d(inputDimension, 64, kernel_size = 7, stride = 1, padding = 3, bias = False),
               torch.nn.BatchNorm2d(64, momentum = BNMOMENTUM),
               torch.nn.ReLU(inplace = True)
           )

def makeResidualPreprocess2x(inputDimension):
    return torch.nn.Sequential(
                torch.nn.Conv2d(inputDimension, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                torch.nn.BatchNorm2d(64, momentum = BNMOMENTUM),
                torch.nn.ReLU(inplace = True)
           )

# for common ResNet, the terminal dimension is 128.
resnetHeatmapTerminal = ResNetTerminal("heatmap", CLASSDIMENSION, 64, heatmapInitializerRes, makeResnetTerminal, process)
resnetSizeTerminal = ResNetTerminal("regr", 4, 64, regressionInitializerRes, makeResnetTerminal, process)
resnetOffsetTerminal = ResNetTerminal("offset", 2, 64, regressionInitializerRes, makeResnetTerminal, process)

class CenterNetResidual(ResNet):

    def __init__(self, numLayers, dims = [64, 64, 128, 256, 512, 256, 256, 256]):
        
        blockType, layers = ResNetSpec[numLayers]
        inputDimension = 1
        super(CenterNetResidual, self).__init__(inputDimension, blockType, layers, 

            # by following the default set of preprocess (4x), we can downsample the resolution
            # from 512x to 128x, decreasing the GPU memory requirement per image.
            # 1x preprocess do not downsample the image, 2x preprocess downsamples by half.

            # preprocess = makeResidualPreprocess1x,
            # preprocess = makeResidualPreprocess2x,
            
            terminals = [resnetHeatmapTerminal, resnetSizeTerminal, resnetOffsetTerminal],
            decoder = decodeCenterNet, dimensions = dims)

        self.initialize(numLayers)

class CenterNetLoss(torch.nn.Module):

    def __init__( self, regressionWeight = 1, offsetWeight = 0.5,
        focal = focalLoss, regression = L1LossMask):

        super(CenterNetLoss, self).__init__()

        self.regressionWeight = regressionWeight
        self.offsetWeight = offsetWeight
        self.focal = focal
        self.regression = regression

    def forward(self, outs, targets):
        heats = []
        regressions = []
        offsets = []
        for out in outs:
            heats += [out["heatmap"]]
            regressions += [reshapeGatherFeatures(out["regr"], targets[3])]
            offsets += [reshapeGatherFeatures(out["offset"], targets[3])]

        groundTruthHeatmap = targets[0]
        groundTruthMask = targets[1]
        groundTruthRegression = targets[2][:,:,2:6]
        groundTruthOffset = targets[2][:,:,0:2]

        # focal loss of heatmap
        focalL = 0

        heats = [clampSigmoid(t) for t in heats]

        focalL += self.focal(heats, groundTruthHeatmap)

        sizeL = 0
        for regr in regressions:
            sizeL += self.regression(regr, groundTruthRegression, groundTruthMask)
        sizeL = self.regressionWeight * sizeL

        offsetL = 0
        for offset in offsets:
            offsetL += self.regression(offset, groundTruthOffset, groundTruthMask)
        offsetL = self.offsetWeight * offsetL

        loss = (focalL + sizeL + offsetL) / len(heats)

        # the loss is one real number, this give a dimension to be a one-element array.
        # torch.tensor([loss]).
        return loss.unsqueeze(0), [focalL, sizeL, offsetL]

def decodeCenterNet( outputDictionary, K = 100, nmsKernelSize = 3, **kwargs ):

    heatmap = outputDictionary["heatmap"]
    regression = outputDictionary["regr"]
    offset = outputDictionary["offset"]

    batch, category, height, width = heatmap.size()

    # apply sigmoid uniform to the heatmap, limiting its value to [0, 1].
    heatmap = torch.sigmoid(heatmap)

    # perform Non-Maximum Suppression (NMS) on heatmaps
    heatmap = nonMaximumSuppression(heatmap, kernelSize = nmsKernelSize)

    # all of these are in format of (batch, k)
    ctScores, ctIndices, ctCategories, ctY, ctX = extractTopK(heatmap, K = K)

    if regression is not None:
        
        # after the reshape feature gathering, the regression now have (batch, k, dim = 2-4)
        regression = reshapeGatherFeatures(regression, ctIndices)

    if offset is not None:
        offset = reshapeGatherFeatures(offset, ctIndices)

    # the output dimensionality are listed as follows:
    # 'ctScores'   : (BATCH, K)
    # 'ctIndices'  : (BATCH, K)
    # 'ctX', 'ctY' : (BATCH, K)
    # 'regression' : (BATCH, K, 4).
    # outputs the objects with top-K scores.

    return [ctScores, ctIndices.long(), ctY.long(), ctX.long(), offset, regression, outputDictionary]

def centerNetEvaluation(xs, ys, ctScores, ctIndices, ctY, ctX, offset, regression, outputDictionary):
    BATCH, K = ctX.shape
    _, MAXTAGLEN, _ = ys[2].shape
    groundTruthHeat = ys[0]
    objNum = [torch.sum(mask).item() for mask in ys[1]]

    # generates the predicted bounding boxes (format to 128x)
    # regression: [majx, majy, minl, halo]

    bounds = torch.zeros(BATCH, K, 4)
    boundsCenter = torch.zeros(BATCH, K, 4)
    boundsOffset = torch.zeros(BATCH, K, 4)

    majL = torch.zeros(BATCH, K)
    majL = torch.sqrt(regression[:,:,0] * regression[:,:,0] + regression[:,:,1] * regression[:,:,1])
    bounds[:,:,0] = ctX - majL + offset[:, :, 0] / 4
    bounds[:,:,1] = ctY - regression[:,:,2] + offset[:, :, 1] / 4
    bounds[:,:,2] = ctX + majL + offset[:, :, 0] / 4
    bounds[:,:,3] = ctY + regression[:,:,2] + offset[:, :, 1] / 4

    boundsCenter[:,:,0] = ctX - 2
    boundsCenter[:,:,1] = ctY - 2
    boundsCenter[:,:,2] = ctX + 2
    boundsCenter[:,:,3] = ctY + 2

    boundsOffset[:,:,0] = ctX - 2 + offset[:, :, 0] / 4
    boundsOffset[:,:,1] = ctY - 2 + offset[:, :, 1] / 4
    boundsOffset[:,:,2] = ctX + 2 + offset[:, :, 0] / 4
    boundsOffset[:,:,3] = ctY + 2 + offset[:, :, 1] / 4

    # calculate the ground-truth from (center-x, center-y, w, h) to (tl-x, tl-y, br-x, br-y).
    groundTruthLocs = torch.zeros(BATCH, MAXTAGLEN, 4)
    groundTruthLocsCenter = torch.zeros(BATCH, MAXTAGLEN, 4)
    groundTruthLocsOffset = torch.zeros(BATCH, MAXTAGLEN, 4)
    if(ys[3].dim() == 2):
        centerY = ys[3] // HEATMAPSIZE                          # [N, MAXTAGLEN]
        centerX = ys[3] - (ys[3] // HEATMAPSIZE) * HEATMAPSIZE  # [N, MAXTAGLEN]
    else:
        centerX = ys[3][:,:,0]
        centerY = ys[3][:,:,1]
    majLP = torch.zeros(BATCH, K)
    majLP = torch.sqrt(ys[2][:,:,2] * ys[2][:,:,2] + ys[2][:,:,3] * ys[2][:,:,3])
    groundTruthLocs[:,:,0] = (centerX - majLP) + ys[2][:,:,0] / 4
    groundTruthLocs[:,:,1] = (centerY - ys[2][:,:,4]) + ys[2][:,:,1] / 4
    groundTruthLocs[:,:,2] = (centerX + majLP) + ys[2][:,:,0] / 4
    groundTruthLocs[:,:,3] = (centerY + ys[2][:,:,4]) + ys[2][:,:,1] / 4

    groundTruthLocsCenter[:,:,0] = (centerX - 2) 
    groundTruthLocsCenter[:,:,1] = (centerY - 2) 
    groundTruthLocsCenter[:,:,2] = (centerX + 2) 
    groundTruthLocsCenter[:,:,3] = (centerY + 2) 

    groundTruthLocsOffset[:,:,0] = (centerX - 2) + ys[2][:,:,0] / 4
    groundTruthLocsOffset[:,:,1] = (centerY - 2) + ys[2][:,:,1] / 4
    groundTruthLocsOffset[:,:,2] = (centerX + 2) + ys[2][:,:,0] / 4
    groundTruthLocsOffset[:,:,3] = (centerY + 2) + ys[2][:,:,1] / 4

    majPred = torch.zeros(BATCH, K, 3) # (x, y, l) for major axis
    majPred[:,:,0] = regression[:,:,0]
    majPred[:,:,1] = regression[:,:,1]
    majPred[:,:,2] = majL

    regrPred = torch.zeros(BATCH, K, 3) # (x, y, l) for major axis
    regrPred[:,:,0] = majL
    regrPred[:,:,1] = regression[:,:,2]
    regrPred[:,:,2] = regression[:,:,3]

    majGt = torch.zeros(BATCH, MAXTAGLEN, 3)
    majGt[:,:,0] = ys[2][:,:,2]
    majGt[:,:,1] = ys[2][:,:,3]
    majGt[:,:,2] = majLP

    regrGt = torch.zeros(BATCH, MAXTAGLEN, 3) # (x, y, l) for major axis
    regrGt[:,:,0] = majLP
    regrGt[:,:,1] = ys[2][:,:,4]
    regrGt[:,:,2] = ys[2][:,:,5]

    if torch.cuda.is_available() and defaultConfig.useGPU:
        bounds = bounds.cuda()
        boundsCenter = boundsCenter.cuda()
        boundsOffset = boundsOffset.cuda()
        groundTruthLocs = groundTruthLocs.cuda()
        groundTruthLocsOffset = groundTruthLocsOffset.cuda()
        groundTruthLocsCenter = groundTruthLocsCenter.cuda()
        majPred = majPred.cuda()
        majGt = majGt.cuda()

        regrPred = regrPred.cuda()
        regrGt = regrGt.cuda()

    # we only selects the predicted boundboxes with a score higher than 0.5 possibility.
    
    validMask = ctScores >= 0.3

    return { 'iouscore': IoUConfidence(bounds, groundTruthLocs, ctScores, validMask),
             'ortho': Orthogonity(bounds, groundTruthLocs, majPred, majGt, validMask),
             'ioucenter': IoU(boundsCenter, groundTruthLocsCenter, validMask),
             'iouoffsetwo': IoU(boundsCenter, groundTruthLocsOffset, validMask),
             'iouoffset': IoU(boundsOffset, groundTruthLocsOffset, validMask),
             'maes': MAE(bounds, groundTruthLocs, regrPred, regrGt, validMask),
             'objs': objNum
            }, outputDictionary