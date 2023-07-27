
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

import PIL
from PIL import Image;
import numpy
import torchvision.transforms as transforms;
import torchvision.transforms.functional as tfunc
import os;
import matplotlib.pyplot as plt

# python utilities ...

import random
from enum import Enum
import math
from configuration import defaultConfig

# the input image of size (1, H, W)  grayscale.

torch.random.manual_seed(42)
numpy.random.seed(42)

@torch.no_grad()
def normalize(tensor) -> torch.Tensor:
    mean = torch.mean(tensor)
    variance = torch.mean(torch.square(tensor - mean))
    tensor = (tensor - mean)/ torch.sqrt(variance)
    return tensor

@torch.no_grad()
def noise(tensor, stdvar = 0.05) -> torch.Tensor:
    _, height, width = tensor.shape
    randomNoise = torch.rand(1, height, width)
    if torch.cuda.is_available() and defaultConfig.useGPU:
        randomNoise = randomNoise.cuda()
    return tensor + (randomNoise - 0.5) * (stdvar * 12)

@torch.no_grad()
def gaussianNoise(tensor, stdvar = 0.05) -> torch.Tensor:
    _, height, width = tensor.shape
    randomNoise = torch.randn(1, height, width)
    if torch.cuda.is_available() and defaultConfig.useGPU:
        randomNoise = randomNoise.cuda()
    return tensor + randomNoise * stdvar

@torch.no_grad()
def varianceJitter(tensor, stdvar = 0.05):
    gaussRand = torch.randn(1)
    if torch.cuda.is_available() and defaultConfig.useGPU:
        gaussRand = gaussRand.cuda() 
    return tensor * (1 + stdvar * gaussRand)

@torch.no_grad()
def horizontalFlip(tensor):
    return torch.flip(tensor, [2])

@torch.no_grad()
def verticalFlip(tensor):
    return torch.flip(tensor, [1])

@torch.no_grad()
def randomFlip(tensor):
    if numpy.random.uniform() > 0.5:
        tensor = torch.flip(tensor, [2])
    if numpy.random.uniform() > 0.5:
        tensor = torch.flip(tensor, [1])
    return tensor

def horizontalFlipNumpy(arr):
    return arr[:, ::-1]

def verticalFlipNumpy(arr):
    return arr[::-1, :]

def randomFlipNumpy(arr):
    if numpy.random.uniform() > 0.5:
        arr = arr[:, ::-1]
    if numpy.random.uniform() > 0.5:
        arr = arr[::-1, :]
    return arr

class PaddingMode(Enum):
    ConstantPadding = 'constant'
    MirrorPadding = 'reflect'
    ReplicatePadding = 'replicate'

class ResampleMode(Enum):
    NearestNeighbour = Image.NEAREST
    Bilinear = Image.BILINEAR
    Bicubic = Image.BICUBIC

# the input image of size (1, 1, H, W)  grayscale.

@torch.no_grad()
def rotateNearestNeighbour(tensor, angle, paddingMode, paddingConstant = 0):
    padding = tensor
    batch, c, height, width = tensor.shape
    paddingRadius = math.sqrt(width ** 2 + height ** 2) / 2
    leftPadding = math.ceil(paddingRadius - 0.5 * width)
    topPadding = math.ceil(paddingRadius - 0.5 * height)

    if paddingMode == PaddingMode.ConstantPadding:
        padding = F.pad(padding, (leftPadding, leftPadding, topPadding, topPadding), 'constant', paddingConstant)
    else:
        padding = F.pad(padding, (leftPadding, leftPadding, topPadding, topPadding), paddingMode.value)
    batch, c, paddingHeight, paddingWidth = padding.shape
    
    # generating the rotation map matrix.
    xs = [-x - 0.5 for x in range(0, int(width/2))][::-1] + [x + 0.5 for x in range(0, int(width/2))]
    ys = [-y - 0.5 for y in range(0, int(height/2))][::-1] + [y + 0.5 for y in range(0, int(height/2))]
    xs = torch.tensor(xs).unsqueeze(0)
    ys = torch.tensor(ys).unsqueeze(1)
    distance = torch.sqrt( torch.pow(xs, 2) + torch.pow(ys, 2) )
    cos = xs / distance
    sin = ys / distance
    sinA = math.sin(angle * math.pi / 180)
    cosA = math.cos(angle * math.pi / 180)

    rotateSin = sin * cosA + cos * sinA
    rotateCos = cos * cosA - sin * sinA
    rotateX = distance * rotateCos + int(width/2) + leftPadding - 0.5
    rotateY = distance * rotateSin + int(height/2) + topPadding - 0.5

    rotateLoc = torch.round(rotateY) * paddingWidth + torch.round(rotateX)
    linePad = padding.reshape(paddingWidth * paddingHeight)
    lineSelector = rotateLoc.reshape(width * height).long()

    gather = torch.gather(linePad, 0, lineSelector).reshape(1, 1, height, width)
    
    return gather

@torch.no_grad()
def rotate(tensor, angle, paddingMode:PaddingMode, resample:ResampleMode, paddingConstant:float = 0):
    padding = tensor.float()
    batch, c, height, width = tensor.shape
    
    paddingRadius = math.sqrt(width ** 2 + height ** 2) / 2
    leftPadding = math.ceil(paddingRadius - 0.5 * width)
    topPadding = math.ceil(paddingRadius - 0.5 * height)
    padding = F.pad(padding, (leftPadding, leftPadding, topPadding, topPadding), paddingMode.value, value = paddingConstant)
    
    rot = tfunc.rotate(padding, angle, resample.value)
    return rot[:,:, topPadding:topPadding + height, leftPadding:leftPadding + width]

@torch.no_grad()
def rotateNonClip(tensor, angle, paddingMode:PaddingMode, resample:ResampleMode, paddingConstant = 0):
    padding = tensor.float()
    batch, c, height, width = tensor.shape
    
    paddingRadius = math.sqrt(width ** 2 + height ** 2) / 2
    leftPadding = math.ceil(paddingRadius - 0.5 * width)
    topPadding = math.ceil(paddingRadius - 0.5 * height)
    padding = F.pad(padding, (leftPadding, leftPadding, topPadding, topPadding), paddingMode.value)
    
    rot = tfunc.rotate(padding, angle, resample.value)
    # return rot[:,:, topPadding:topPadding + height, leftPadding:leftPadding + width]
    return rot, leftPadding, topPadding

def randomRotate(tensor, paddingMode:PaddingMode, resample:ResampleMode, paddingConstant = 0):
    return rotate(tensor, numpy.random.uniform() * 90, paddingMode, resample, paddingConstant)