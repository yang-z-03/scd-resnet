
# pytorch references ...
# and pytorch multi-gpu support

from re import T
import torch;
import torch.autograd
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
from models.backbones.cornerPooling import TopPool, BottomPool, LeftPool, RightPool
from models.backbones.residuals import BNMOMENTUM, ResNet, ResNetSpec, ResNetTerminal, Residual

from models.backbones.utility import clampSigmoid, convolutionConv1x1, extractTopK, nonMaximumSuppression, reshapeGatherFeatures
from models.losses.focal import focalLoss
from models.losses.regression import smoothL1LossMask
from models.backbones.stackHourglass import HourglassTerminal, StackHourglass
from evaluations.detection import averageIoU, averagePrecision
from configuration import defaultConfig
from datasets.confocalCenters.confocalCenter import DOWNSAMPLE

CLASSDIMENSION = 3

# a simplified CenterNet for object detection consists of one center gaussian heatmap
# branch and a regression branch for object size. this implementation do not use the 
# offset regression branch, since the samples is used at its original resolution.

# the corner pool gathers the information from graphic features to infer the location
# of bounding points. a line/column maximum pool.
#
# after the prediction network redimension, apply this corner pooling layers.
#
#                             [(N, 'predictionDim', H, W)]
#                                          |
#                 ,------------------------+------------------------,
#         branch1 |                branch2 |                        |
#                 v                        v                        | shortcut conv
#         [(N, 128, H, W)]          [(N, 128, H, W)]                v
#                 |                        |           [(N, 'predictionDim', H, W)]
# branch pooling1 |       branch pooling 2 |                        |
#                 v                        v                        | bn
#         [(N, 128, H, W)]          [(N, 128, H, W)]                |
#                 |                        |                        |
#                 '----------> + <---------'                        |
#                              |                                    |
#                              | branch merge                       |
#                              v                                    |
#                 [(N, 'predictionDim', H, W)]                      |
#                              |                                    |
#                              | branch merge bn                    |
#                              '----------------> + <---------------'
#                                                 |
#                                                 | ReLU
#                                                 | lastConv
#                                                 v
#                                    [(N, 'predictionDim', H, W)]

class CornerPool(torch.nn.Module):

    def __init__(self, predictionDimension, pool1, pool2):
        super(CornerPool, self).__init__()

        self.branch1 = Convolution(3, predictionDimension, 128)
        self.branch2 = Convolution(3, predictionDimension, 128)

        self.branchMerge = torch.nn.Conv2d(128, predictionDimension, (3, 3), padding=(1, 1), bias=False)
        self.branchMergeBn = torch.nn.BatchNorm2d(predictionDimension)

        self.shortcutConv = torch.nn.Conv2d(predictionDimension, predictionDimension, (1, 1), bias = False)
        self.shortcutBn = torch.nn.BatchNorm2d(predictionDimension)
        self.mixReLU = torch.nn.ReLU(inplace = True)

        self.lastConv = Convolution(3, predictionDimension, predictionDimension)

        self.branchPooling1 = pool1()
        self.branchPooling2 = pool2()

    def forward(self, x):
        
        # pool 1
        p1_conv1 = self.branch1(x)
        pool1    = self.branchPooling1(p1_conv1)

        # pool 2
        p2_conv1 = self.branch2(x)
        pool2    = self.branchPooling2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.branchMerge(pool1 + pool2)
        p_bn1   = self.branchMergeBn(p_conv1)

        conv1 = self.shortcutConv(x)
        bn1   = self.shortcutBn(conv1)
        relu1 = self.mixReLU(p_bn1 + bn1)

        conv2 = self.lastConv(relu1)
        return conv2

class TopLeftPool(CornerPool):
    def __init__(self, dim):
        super(TopLeftPool, self).__init__(dim, TopPool, LeftPool)

class BottomRightPool(CornerPool):
    def __init__(self, dim):
        super(BottomRightPool, self).__init__(dim, BottomPool, RightPool)

def makeTopLeftLayer(dim):
    return TopLeftPool(dim)

def makeBottomRightLayer(dim):
    return BottomRightPool(dim)

def makePoolLayer(dim):
    return torch.nn.Sequential()

def process(inp, module, *xs, **kwargs):
    return module(inp)

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

    return layer

def makeTopLeftTerminal(prediction, current, output):

    if current > 0:
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(prediction, current, kernel_size = 3, padding = 1, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(current, output, kernel_size = 1, stride = 1, padding = 0))
    else:
        layer = torch.nn.Conv2d (
                    in_channels     = prediction,
                    out_channels    = output,
                    kernel_size     = 1,
                    stride          = 1,
                    padding         = 0
                )

    return layer

def makeBottomRightTerminal(prediction, current, output):

    if current > 0:
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(prediction, current, kernel_size = 3, padding = 1, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(current, output, kernel_size = 1, stride = 1, padding = 0))
    else:
        layer = torch.nn.Conv2d (
                    in_channels     = prediction,
                    out_channels    = output,
                    kernel_size     = 1,
                    stride          = 1,
                    padding         = 0
                )

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

class CornerNetResidual(ResNet):

    def __init__(self, numLayers):
        
        blockType, layers = ResNetSpec[numLayers]
        inputDimension = 1
        super(CornerNetResidual, self).__init__(inputDimension, blockType, layers, 

            # by following the default set of preprocess, we can downsample the resolution
            # from 512x to 128x, decreasing the GPU memory requirement per image.
            # preprocess = makeResidualPreprocess,
            
            terminals = [resnetHeatmapTerminal],
            decoder = decodeCornerNet )

        self.initialize(numLayers)

class CornerNetLoss(torch.nn.Module):

    def __init__( self, focal = focalLoss):

        super(CornerNetLoss, self).__init__()

        self.focal = focal

    #@torch.autograd.anomaly_mode.set_detect_anomaly(True)
    def forward(self, outs, targets):
        heats = []
        tlheats = []
        brheats = []
        for out in outs:
            N, D, H, W = out['heatmap'].shape
            heats += [out['heatmap'][:, 0, :, :].clone().unsqueeze(1)]
            tlheats += [out['heatmap'][:, 1, :, :].clone().unsqueeze(1)]
            brheats += [out['heatmap'][:, 2, :, :].clone().unsqueeze(1)]

        groundTruthHeatmap = targets[0]
        groundTruthTL = targets[3]
        groundTruthBR = targets[4]

        # focal loss of heatmap
        focalL = 0

        heats = [clampSigmoid(t) for t in heats]
        tlheats = [clampSigmoid(t) for t in tlheats]
        brheats = [clampSigmoid(t) for t in brheats]

        focalL += self.focal(heats, groundTruthHeatmap)
        focalL += self.focal(tlheats, groundTruthTL)
        focalL += self.focal(brheats, groundTruthBR)

        loss = (focalL) / len(heats)

        # the loss is one real number, this give a dimension to be a one-element array.
        # torch.tensor([loss]).
        return loss.unsqueeze(0), {}

def decodeCornerNet( outputDictionary, K = 100, nmsKernelSize = 3 ):

    heatmap = outputDictionary["heatmap"][:, 0, :, :].unsqueeze(1)
    tl = outputDictionary["heatmap"][:, 1, :, :].unsqueeze(1)
    br = outputDictionary["heatmap"][:, 2, :, :].unsqueeze(1)

    batch, category, height, width = heatmap.size()

    # apply sigmoid uniform to the heatmap, limiting its value to [0, 1].
    heatmap = torch.sigmoid(heatmap)
    tl = torch.sigmoid(tl)
    br = torch.sigmoid(br)

    # perform Non-Maximum Suppression (NMS) on heatmaps
    heatmap = nonMaximumSuppression(heatmap, kernelSize = nmsKernelSize)
    tl = nonMaximumSuppression(tl, kernelSize = nmsKernelSize)
    br = nonMaximumSuppression(br, kernelSize = nmsKernelSize)

    # all of these are in format of (batch, k)
    ctScores, ctIndices, ctCategories, ctY, ctX = extractTopK(heatmap, K = K)
    tlScores, tlIndices, tlCategories, tlY, tlX = extractTopK(tl, K = K)
    brScores, brIndices, brCategories, brY, brX = extractTopK(br, K = K)

    # the output dimensionality are listed as follows:
    # 'ctScores'   : (BATCH, K)
    # 'ctIndices'  : (BATCH, K)
    # 'ctX', 'ctY' : (BATCH, K)
    # outputs the objects with top-K scores.

    return [ ctScores, ctIndices.long(), ctY.long(), ctX.long(), 
             tlScores, tlIndices.long(), tlY.long(), tlX.long(),
             brScores, brIndices.long(), brY.long(), brX.long(),
             outputDictionary]

def cornerNetEvaluation(xs, ys, ctScores, ctIndices, ctY, ctX, tlScores, tlIndices, tlY, tlX, \
                        brScores, brIndices, brY, brX, outputDictionary):
    BATCH, K = ctX.shape
    _, MAXTAGLEN, _ = ys[2].shape
    groundTruthHeat = ys[0]
    groundTruthTL = ys[3]
    groundTruthBR = ys[4]
    objNum = [torch.sum(mask).item() for mask in ys[1]]

    return { 'heatAP50': averagePrecision(ctX, ctY, groundTruthHeat, objNum, 0.5),
             'heatAP75': averagePrecision(ctX, ctY, groundTruthHeat, objNum, 0.75),
             'tlAP50': averagePrecision(tlX, tlY, groundTruthTL, objNum, 0.5),
             'tlAP75': averagePrecision(tlX, tlY, groundTruthTL, objNum, 0.75),
             'brAP50': averagePrecision(brX, brY, groundTruthBR, objNum, 0.5),
             'brAP75': averagePrecision(brX, brY, groundTruthBR, objNum, 0.75)
           }, outputDictionary