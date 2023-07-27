
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

class Convolution(torch.nn.Module):

    # initialize the convolution layer with the given parameters.
    # apply a convolution with size * size to the input with shape (N, 'inputDimension', H, W), and 
    # returns output with shape (N, 'outputDimension', H, W). with a padding of zero attached to
    # its border to ensure the input and output are of the same size.
    # 
    # the batch normalization by default is the unparallelled version, call conversion helper
    # method `torch.nn.SyncBatchNorm.convert_sync_batchnorm` for module conversion.

    def __init__(self, convSize, inputDimension, outputDimension, stride = 1, batchNorm = True):
        
        super(Convolution, self).__init__()

        pad = (convSize - 1) // 2
        self.conv = torch.nn.Conv2d(inputDimension, outputDimension, (convSize, convSize),
            padding = (pad, pad), stride = (stride, stride), bias = not batchNorm)
        self.bn = torch.nn.BatchNorm2d(outputDimension) if batchNorm else torch.nn.Sequential()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class FullyConnected(torch.nn.Module):

    # initialize the fully-connected layer with input shape (N, 'inputElements') 2d tensor
    # and output shape of (N, 'outputElements') tensor.

    def __init__(self, inputElements, outputElements, batchNorm = True):

        super(FullyConnected, self).__init__()

        self.withBatchNorm = batchNorm

        self.linear = torch.nn.Linear(inputElements, outputElements)
        if self.withBatchNorm:
            self.bn = torch.nn.BatchNorm1d(outputElements)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.withBatchNorm else linear
        relu   = self.relu(bn)
        return relu