
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
from models.backbones.pooling import poolingLayer, unpoolingLayer
from models.backbones.residuals import Residual, residual3x3
from models.backbones.terminal import BackboneTerminal
from models.backbones.utility import \
    changeDimensionConv, clampSigmoid, convolutionConv1x1, extractTopK, gatherFeatures, mergeLayer, noLayer, \
    nonMaximumSuppression, reshapeGatherFeatures, stackLayers, stackLayersReverted
from models.backbones.hourglass import Hourglass
from models.losses.embeddings import embeddingLoss
from models.losses.focal import focalLoss
from configuration import defaultConfig
from logger import Logger
import sys

class HourglassTerminal(BackboneTerminal):

    # interface of hourglass terminal. optional implement the initializer function
    # >> func (model:torch.nn.Module)
    # and the makeLayer function
    # >> func (predictionDimension:int, currentDimension:int, outputDimension:int)

    def __init__(self, name, outputDimension, 
        initializerFunction = None, makeLayerFunction = None, process = None):

        super(HourglassTerminal, self).__init__(name, initializerFunction, makeLayerFunction, process)

        self.outputDimension = outputDimension

class StackHourglass(torch.nn.Module):

    # stacked hourglass network is one of the state-of-the-art object detection networks.
    # it considers images of different resolution and output through its terminals.
    # the network architecture is demonstrated as follows:
    #
    #           [ preprocess (B, 3, H, W) ]
    #                       |
    #                       | Conv(7, 2) [3 -> 128]
    #           .preprocess | Residual(3, 2) [128 -> 'currentDimension']
    #                       v
    #           [ backbone input (B, 'currentDimension', H/4, W/4) ] -----------,
    #                       |                                                   |
    #                       | Hourglass ['currentDimension' ->                  |
    #                       |            'currentDimension']                    |
    #                       v                                                   |
    #           [ kp (B, 'currentDimension', H/4, W/4) ]                        |
    #                       |                                                   |
    #                       | Redim. Convolution ['currentDimension'            | Shortcut
    #                       | -> 'predictionDimension']                         | Layers
    #                       v                                                   | ['currentDimension' ->
    #           [ cnv (B, 'predictionDimension', H/4, W/4) ]                    |  'currentDimension']
    #                       |                                                   |
    #                       |                                                   |
    #         terminals <---|                                                   |
    #  ['prediction' ->     |                                                   |
    #   'current']          | Convolution Previous Hourglass                    |
    #                       | ['predictionDimension' ->                         |
    #                       |  'currentDimension']                              |
    #                       v                                                   |
    #                       + <-------------------------------------------------'
    #                       |
    #                       | relu
    #                       v
    #            [ inter1 (B, 'currentDimension', H/4, W/4) ]
    #                       |
    #                       | Inter-Hourglass Layers
    #                       | ['currentDimension' -> 'currentDimension']
    #                       v 
    #            [ backbone input (B, 'currentDimension', H/4, W/4)]
    #                       |
    #                       ...

    # the output format of this network is specified in the 'terminals' parameters,
    # it is encoded by a list of possible terminal types, for example, heatmap or regression.

    def __init__(
        self, hourglassIteration, hourglassStacks, dimensions, modules, outputDimension,
        beforeBackbone = None, predictionConvDim = 256, 

        makeConvolutionLayer = changeDimensionConv,

        hourglassBeforeDownsample = stackLayers, hourglassCentral = stackLayers, 
        hourglassBefore = stackLayers, hourglassAfter = stackLayersReverted,
        hourglassPool = poolingLayer, hourglassUnpooling = unpoolingLayer,
        hourglassMerge = mergeLayer, makeInternalLayer = residual3x3, 
        hourglassLayer = Residual, decoder = None,
        
        terminals = [] ):

        super(StackHourglass, self).__init__()

        self.hourglassStacks = hourglassStacks
        self.decoder = decoder

        currentDimension = dimensions[0]

        self.preprocess = torch.nn.Sequential(
            Convolution ( 7, 3, 128, stride = 2 ),
            # Residual ( 3, 128, 256, stride = 2 )
            Residual ( 3, 128, currentDimension, stride = 2 ) # CHANGE
        ) if beforeBackbone is None else beforeBackbone

        self.hourglassStack  = torch.nn.ModuleList([
            Hourglass(
                # FIX: the dimensions have one element left?
                hourglassIteration, dimensions, modules, layer = hourglassLayer,
                layersBeforeDownsample = hourglassBeforeDownsample,
                layersCentral = hourglassCentral,
                layersBeforeHourglass = hourglassBefore,
                layersAfterHourglass = hourglassAfter,
                layersDownsampling = hourglassPool,
                layersUpsampling = hourglassUnpooling,
                layersMerge = hourglassMerge
            ) for _ in range(hourglassStacks)
        ])

        # the output size of the hourglass backbone has 'currentDimension', and this 
        # reshapes the output to 'predictionConvDim'.
        self.redimConvolution = torch.nn.ModuleList([
            makeConvolutionLayer(currentDimension, predictionConvDim) for _ in range(hourglassStacks)
        ])

        self.terminalLayers = {}
        self.terminals = {}
        for terminal in terminals:
            module = torch.nn.Sequential(*[
                terminal.makeLayer(predictionConvDim, currentDimension, terminal.outputDimension) \
                for _ in self.hourglassStack
            ]) if terminal.makeLayer is not None else torch.nn.Sequential(*[torch.nn.Sequential() for _ in self.hourglassStack])

            if torch.cuda.is_available() and defaultConfig.useGPU:
                module = module.cuda()
            
            if terminal.initializer is not None:
                for layer in module:
                    terminal.initializer(layer)
            self.terminalLayers[terminal.name] = module
            self.terminals[terminal.name] = terminal
            setattr(self, terminal.name, module)

        # layers between the stacked hourglass.
        self.interHourglassLayers = torch.nn.ModuleList([
            makeInternalLayer(currentDimension) for _ in range(hourglassStacks - 1)
        ])

        self.shortcutLayers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(currentDimension, currentDimension, (1, 1), bias = False),
                torch.nn.BatchNorm2d(currentDimension)
            ) for _ in range(hourglassStacks - 1)
        ])

        self.convPrevHourglass = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(predictionConvDim, currentDimension, (1, 1), bias = False),
                torch.nn.BatchNorm2d(currentDimension)
            ) for _ in range(hourglassStacks - 1)
        ])

        self.relu = torch.nn.ReLU(inplace = True)

    # the architecture is drawn above.
    def trainNetwork(self, *args, **kwargs):
        image   = args[0]

        # prepare the image from an original resolution to those fitted with the 
        # hourglass backbone's input.
        inter = self.preprocess(image)
        outs  = []

        layers = zip(
            self.hourglassStack, self.redimConvolution,
        )

        for id, layer in enumerate(layers):
            hourglass, redimConv = layer[0:2]

            kp  = hourglass(inter)
            cnv = redimConv(kp)

            out = {}
            for terminal in self.terminalLayers.keys():
                if self.terminals[terminal].process is not None:
                    out[terminal] = self.terminals[terminal].process(cnv, getattr(self, terminal), *args)
                else:
                    Logger.err("Processor function of the terminal '{}' is not implemented.".format(terminal))
                    sys.exit()

            outs += [out]

            if id < self.hourglassStacks - 1:
                inter = self.shortcutLayers[id](inter) + self.convPrevHourglass[id](cnv)
                inter = self.relu(inter)
                inter = self.interHourglassLayers[id](inter)

        return outs

    def evalNetwork(self, *xs, **kwargs):
        image = xs[0]

        inter = self.preprocess(image)
        outs  = []

        layers = zip(
            self.hourglassStack, self.redimConvolution
        )

        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.hourglassStacks - 1:
                out = {}
                for terminal in self.terminalLayers.keys():
                    if self.terminals[terminal].process is not None:
                        out[terminal] = self.terminals[terminal].process(cnv, self.terminalLayers[terminal][ind], *xs, **kwargs)
                    else:
                        Logger.err("Processor function of the terminal '{}' is not implemented.".format(terminal))
                        sys.exit()

                outs += [out]

            if ind < self.hourglassStacks - 1:
                inter = self.shortcutLayers[ind](inter) + self.convPrevHourglass[ind](cnv)
                inter = self.relu(inter)
                inter = self.interHourglassLayers[ind](inter)

        # decode the output features in the dictionary. with keys in the name of each branch.
        return self.decoder(outs[0], **kwargs)

    def forward(self, *xs, **kwargs):
        decode = False
        if 'decode' in kwargs: decode = kwargs['decode']

        if not decode:
            return self.trainNetwork(*xs, **kwargs)
        return self.evalNetwork(*xs, **kwargs)
