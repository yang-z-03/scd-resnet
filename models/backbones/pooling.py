
# pytorch references ...
# and pytorch multi-gpu support

import imp
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

from enum import Enum

class PoolingType(Enum):
    MaximalPool = 0
    AveragePool = 2

class UpsampleType(Enum):
    NearestNeighbour = 'nearest'
    Linear = 'linear'
    Bilinear = 'bilinear'
    Trilinear = 'trilinear'
    Bicubic = 'bicubic'

def poolingLayer(scaleFactor = 2, downsampleType:PoolingType = PoolingType.MaximalPool,
                 width:int = None, height:int = None):
    if downsampleType == PoolingType.MaximalPool:
        return torch.nn.MaxPool2d(kernel_size = scaleFactor, stride = scaleFactor)
    elif downsampleType == PoolingType.AveragePool:
        return torch.nn.AvgPool2d(kernel_size = scaleFactor, stride = scaleFactor)

def adaptivePoolingLayer(outputWidth, outputHeight, downsampleType:PoolingType = PoolingType.MaximalPool):
    if downsampleType == PoolingType.MaximalPool:
        return torch.nn.AdaptiveMaxPool2d((outputHeight, outputWidth))
    elif downsampleType == PoolingType.AveragePool:
        return torch.nn.AdaptiveAvgPool2d((outputHeight, outputWidth))

def unpoolingLayer(scaleFactor = 2, upsampleType:UpsampleType = UpsampleType.NearestNeighbour):
    return torch.nn.Upsample(scale_factor = scaleFactor, mode = upsampleType.value)