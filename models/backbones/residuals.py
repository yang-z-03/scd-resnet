
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
import sys

from logger import Logger
from models.backbones.convolutions import Convolution
from models.backbones.terminal import BackboneTerminal
from models.backbones.utility import convolution3x3
from configuration import defaultConfig

BNMOMENTUM = 0.1

class Residual(torch.nn.Module):

    # initialize the residual block (by K. He's ResNet) with the following sequence of components:
    #   
    #   module names                   data shape
    # ============================================================================================
    # - convolution (3 x 3, pad 1)     (N, 'inputDimension', H, W) -> (N, 'outputDimension', H, W)
    # - batch normalization            -
    # - ReLU                           -
    # - convolution (3 x 3, pad 1)     (N, 'outputDimension', H, W) -> (N, 'outputDimension', H, W)
    # - batch normalization            -
    #
    # together with a skip route with a convolution and batch normalization to resize a copy of
    # the input data, and combine the two route with relu(conv + skip).

    # note that the 'convSize' and 'batchNorm' is not used, but as a placeholder parameter for
    # higher abstract levels' format of initialization.

    def __init__(self, convSize, inputDimension, outputDimension, stride = 1, batchNorm = True):

        super(Residual, self).__init__()

        self.conv1 = torch.nn.Conv2d(inputDimension, outputDimension, (3, 3), padding = (1, 1), stride = (stride, stride), bias = False)
        self.bn1   = torch.nn.BatchNorm2d(outputDimension)
        self.relu1 = torch.nn.ReLU(inplace = True)

        self.conv2 = torch.nn.Conv2d(outputDimension, outputDimension, (3, 3), padding = (1, 1), bias = False)
        self.bn2   = torch.nn.BatchNorm2d(outputDimension)
        
        self.skip  = torch.nn.Sequential(
            torch.nn.Conv2d(inputDimension, outputDimension, (1, 1), stride = (stride, stride), bias = False),
            torch.nn.BatchNorm2d(outputDimension)
        ) if stride != 1 or inputDimension != outputDimension else torch.nn.Sequential()
        self.relu  = torch.nn.ReLU(inplace = True)

    def forward(self, x):

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def residual3x3(dimension):
    return Residual(3, dimension, dimension)

class BasicBlock(torch.nn.Module):
    
    expansion = 1

    def __init__(self, inputDimension, outputDimension, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()

        self.conv1 = convolution3x3(inputDimension, outputDimension, stride)
        self.bn1 = torch.nn.BatchNorm2d(outputDimension, momentum = BNMOMENTUM)
        self.relu = torch.nn.ReLU(inplace = True)
        self.conv2 = convolution3x3(outputDimension, outputDimension)
        self.bn2 = torch.nn.BatchNorm2d(outputDimension, momentum = BNMOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        # the residual is the shortcut path to the originals. can be dowmsampled to
        # lower resolution if downsample is available. to keep resolution the same, stride
        # should be set if downsample is available.

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(torch.nn.Module):
    
    # the expansion is the ratio of central convolution dimension to the target
    # output dimension. this bottleneck network gives an output of 4 * outputDimension.
    expansion = 4

    def __init__(self, inputDimension, outputDimension, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()

        self.conv1 = torch.nn.Conv2d(inputDimension, outputDimension, kernel_size = 1, bias = False)
        self.bn1 = torch.nn.BatchNorm2d(outputDimension, momentum = BNMOMENTUM)
        self.conv2 = torch.nn.Conv2d(outputDimension, outputDimension, kernel_size = 3, 
                                     stride = stride, padding = 1, bias = False)
        self.bn2 = torch.nn.BatchNorm2d(outputDimension, momentum = BNMOMENTUM)
        self.conv3 = torch.nn.Conv2d(outputDimension, outputDimension * self.expansion, 
                                     kernel_size = 1, bias = False)
        self.bn3 = torch.nn.BatchNorm2d(outputDimension * self.expansion,
                                        momentum = BNMOMENTUM)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetTerminal(BackboneTerminal):

    # interface of hourglass terminal. optional implement the initializer function
    # >> func (model:torch.nn.Module)
    # the makeLayer function
    # >> func (predictionDimension:int, currentDimension:int, outputDimension:int)
    # and the process function
    # >> func (input:Tensor, *xs, **kwargs)

    def __init__(self, name, outputDimension, terminalDimension = 0, 
        initializerFunction = None, makeLayerFunction = None, process = None):

        super(ResNetTerminal, self).__init__(name, initializerFunction, makeLayerFunction, process)

        self.outputDimension = outputDimension
        self.terminalDimension = terminalDimension

class ResNet(torch.nn.Module):

    # the 'block' and 'layers' parameters are specified by the ResNet Specification.
    # 
    # {  18: (BasicBlock, [2, 2,  2, 2] ),
    #    34: (BasicBlock, [3, 4,  6, 3] ),
    #    50: (Bottleneck, [3, 4,  6, 3] ),
    #   101: (Bottleneck, [3, 4, 23, 3] ),
    #   152: (Bottleneck, [3, 8, 36, 3] )}

    # the 'dimensions' parameters are used to define inner network dimensions of a ResNet. with default values
    # dimensions       [0] : 64                : dimensions of preprocessor
    #            [1] ~ [4] : 64, 128, 256, 512 : dimensions of resnet layers 1 ~ 4
    #            [5] ~ [6] : 256, 256          : dimensions of first two upsampling layer
    #                  [7] : 256               : dimension of prediction output layer

    def __init__(self, inputDimension, block, layers, preprocess = None, terminals = [], decoder = None, 
                 dimensions = [64, 64, 128, 256, 512, 256, 256, 256], **kwargs):
        
        self.inputDimension = dimensions[0]
        self.deconvolutionWithBias = False
        self.terminals = {}
        self.decoder = decoder

        super(ResNet, self).__init__()
        if preprocess is None:
            self.preprocess = torch.nn.Sequential(
                torch.nn.Conv2d(inputDimension, dimensions[0], kernel_size = 7, stride = 2, padding = 3, bias = False),
                torch.nn.BatchNorm2d(dimensions[0], momentum = BNMOMENTUM),
                torch.nn.ReLU(inplace = True),
                torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            ) 
        else: self.preprocess = preprocess(inputDimension)

        self.layer1 = self.makeLayer(block, dimensions[1], layers[0])
        self.layer2 = self.makeLayer(block, dimensions[2], layers[1], stride = 2)
        self.layer3 = self.makeLayer(block, dimensions[3], layers[2], stride = 2)
        self.layer4 = self.makeLayer(block, dimensions[4], layers[3], stride = 2)

        self.prediction = dimensions[7]
        self.deconvolutionLayers = self.makeDeconvLayer(
            3,
            [dimensions[5], dimensions[6], self.prediction],
            [4, 4, 4],
        )

        self.terminalLayers = {}
        for terminal in terminals:
            outputDim = terminal.outputDimension
            terminalDim = terminal.terminalDimension

            if terminal.makeLayer is not None:
                terminalLayer = terminal.makeLayer(self.prediction, terminalDim, outputDim)
            else:
                terminalLayer = torch.nn.Conv2d (
                    in_channels     = self.prediction,
                    out_channels    = outputDim,
                    kernel_size     = 1,
                    stride          = 1,
                    padding         = 0
                )
            
            if torch.cuda.is_available() and defaultConfig.useGPU:
                terminalLayer = terminalLayer.cuda()
            
            self.terminals[terminal.name] = terminal
            self.terminalLayers[terminal.name] = terminalLayer
    
        for terminal in terminals:
            setattr(self, terminal.name, self.terminalLayers[terminal.name])

    # the 'blocks' parameter indicates the number of repeated block elements in the layer.
    def makeLayer(self, block, dimension, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inputDimension != dimension * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inputDimension, dimension * block.expansion,
                                kernel_size = 1, stride = stride, bias = False),
                torch.nn.BatchNorm2d(dimension * block.expansion, momentum = BNMOMENTUM),
            )

        layers = []
        layers.append(block(self.inputDimension, dimension, stride, downsample))
        self.inputDimension = dimension * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inputDimension, dimension))

        return torch.nn.Sequential(*layers)

    def getDeconvConfig(self, kernel, index):
        if kernel == 4:
            padding = 1
            outputPadding = 0
        elif kernel == 3:
            padding = 1
            outputPadding = 1
        elif kernel == 2:
            padding = 0
            outputPadding = 0

        return kernel, padding, outputPadding

    def makeDeconvLayer(self, nLayers, dimensions, kernels):

        if nLayers != len(dimensions) or nLayers != len(kernels):
            Logger.err(":: residuals.py :: Inconsistant Number of Layers. ")
            sys.exit()

        numLayers = []
        for i in range(nLayers):
            kernel, padding, outputPadding = self.getDeconvConfig(kernels[i], i)

            dimension = dimensions[i]
            numLayers.append(
                torch.nn.ConvTranspose2d(
                    in_channels     = self.inputDimension,
                    out_channels    = dimension,
                    kernel_size     = kernel,
                    stride          = 2,
                    padding         = padding,
                    output_padding  = outputPadding,
                    bias            = self.deconvolutionWithBias ))
            numLayers.append(torch.nn.BatchNorm2d(dimension, momentum = BNMOMENTUM))
            numLayers.append(torch.nn.ReLU(inplace = True))
            self.inputDimension = dimension

        return torch.nn.Sequential(*numLayers)

    def forward(self, *x, **kwargs):
        decode = False
        if 'decode' in kwargs: decode = kwargs['decode']

        input = x[0]

        input = self.preprocess(input)

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = self.deconvolutionLayers(input)
        ret = {}
        for head in self.terminalLayers.keys():
            if self.terminals[head].process is not None:
                ret[head] = self.terminals[head].process(input, getattr(self, head), *x, **kwargs)
            else:
                Logger.err("Processor function of the terminal '{}' is not implemented.".format(head))
                sys.exit()

        return [ret] if not decode else self.decoder(ret)

    def initialize(self, num_layers):

        for _, m in self.deconvolutionLayers.named_modules():
            if isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, std=0.001)
                if self.deconvolutionWithBias:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

            for head in self.terminalLayers.keys():
              terminal = self.terminalLayers[head]
              for i, m in enumerate(terminal.modules()):
                  if isinstance(m, torch.nn.Conv2d):
                      if m.weight.shape[0] == self.terminals[head].outputDimension:
                          if self.terminals[head].initializer is not None:
                              self.terminals[head].initializer(m)

ResNetSpec  = {  18: (BasicBlock, [2, 2,  2, 2] ),
                 34: (BasicBlock, [3, 4,  6, 3] ),
                 50: (Bottleneck, [3, 4,  6, 3] ),
                101: (Bottleneck, [3, 4, 23, 3] ),
                152: (Bottleneck, [3, 8, 36, 3] ),
              
                 16: (BasicBlock, [1, 2,  2, 2] ),
                 14: (BasicBlock, [1, 2,  2, 1] ),
                 12: (BasicBlock, [1, 1,  2, 1] ),
                 10: (BasicBlock, [1, 1,  1, 1] )
              }