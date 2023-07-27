
# pytorch references ...
# and pytorch multi-gpu support

import imp
from inspect import stack
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

from models.backbones.residuals import Residual
from models.backbones.utility import stackLayers, stackLayersReverted, mergeLayer
from models.backbones.pooling import poolingLayer, unpoolingLayer

class Hourglass(torch.nn.Module):

    # initialize the hourglass network using the following parameters:
    # iterations - the number of cascade hourglass network in the module. for example, if
    #              'iteration' is set to 3, the whole network will look like:
    #              (iteration = 3 hourglass:
    #                  ...
    #                  (iteration = 2 hourglass:
    #                      (layersBeforeDownsample:
    #                          dimensions[1] > dimensions[1] : residual (stack = modules[1]))
    #                      (layersDownsampling:
    #                          down-sampling layer)
    #                      (layersBeforeHourglass:
    #                          dimensions[1] > dimensions[2] : residual (stack = modules[1]))
    #                      
    #                      (iteration = 1 stack:
    #                          dimensions[2] > dimensions[2] : residual (stack = modules[2])
    #                      )
    #
    #                      (layersAfterHourglass:
    #                          dimensions[2] > dimensions[1] : residual (stack = modules[1]))
    #                  )
    #              )
    # dimensions - an array of length 'iteration' to specify the dimensions at each
    #              hourglass layer. the last one in the array is the central dimension.
    # modules - the number of stacked residual blocks (or other types of layer specified
    #           by the 'layer' class) in each hourglass. an array of length 'iteration'.
    # layer - the basic building blocks of every layer in the network, residual by default.
    # layers* - the functions generating layers at different places in the network.

    def __init__(
        self, iterations, dimensions, modules, layer = Residual,
        layersBeforeDownsample = stackLayers, layersCentral = stackLayers,
        layersBeforeHourglass = stackLayers, layersAfterHourglass = stackLayersReverted,
        layersDownsampling = poolingLayer, layersUpsampling = unpoolingLayer,
        layersMerge = mergeLayer, **kwargs ):

        super(Hourglass, self).__init__()

        self.iteration = iterations

        currentModule = modules[0]
        nextModule = modules[1]

        currentDimension = dimensions[0]
        nextDimension = dimensions[1]

        self.preserveCurrentDimension  = layersBeforeDownsample(
            3, currentDimension, currentDimension, currentModule, layer = layer, **kwargs )

        self.downSampling = layersDownsampling(2)

        self.changeDimension = layersBeforeHourglass(
            3, currentDimension, nextDimension, currentModule, layer = layer, **kwargs )

        if self.iteration > 1:
            self.embeddedHourglass = Hourglass(
                iterations - 1, dimensions[1:], modules[1:], layer = layer, 
                layersBeforeDownsample = layersBeforeDownsample, 
                layersCentral = layersCentral,
                layersBeforeHourglass = layersBeforeHourglass,
                layersAfterHourglass = layersAfterHourglass,
                layersDownsampling = layersDownsampling,
                layersUpsampling = layersUpsampling,
                layersMerge = layersMerge, **kwargs ) 
        else:
            self.embeddedHourglass = layersCentral( 3, nextDimension, nextDimension, nextModule, layer = layer, **kwargs )
        
        self.changeDimensionBack = layersAfterHourglass(
            3, nextDimension, currentDimension, currentModule,
            layer=layer, **kwargs )

        self.upSampling  = layersUpsampling(2)

        self.merge = layersMerge(currentDimension)

    def forward(self, x):
        up1  = self.preserveCurrentDimension(x)
        max1 = self.downSampling(x)
        low1 = self.changeDimension(max1)
        low2 = self.embeddedHourglass(low1)
        low3 = self.changeDimensionBack(low2)
        up2  = self.upSampling(low3)
        return self.merge(up1, up2)
