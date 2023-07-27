
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

# calculate the corner gaussian radius given an IoU threshold.
# for the largest available prediction, the intersection will be 
# $$ I = (w - 2r\cos{\theta})(h - 2r\sin{\theta}) $$ over the union of area $ wh $. 
# thus, $$ IoU = \frac{(w - 2r\cos{\theta})(h - 2r\sin{\theta})}{wh} $$ where
# $$ \sin{\theta} = \frac{w}{\sqrt{w^2 + h^2}} $$.

def cornerThresholdRadius(width, height, threshold = 0.7):
    sumSq = width ** 2 + height ** 2
    prod = width * height
    radius = ((2 * sqrt(sumSq)/prod) - sqrt(4 * sumSq / (prod ** 2) - (16 * (1 - threshold))/sumSq)) / (8 / sumSq)
    return radius

def centerThresholdRadius(width, height, threshold=0.7):
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - threshold) / (1 + threshold)
    sq1 = numpy.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - threshold) * width * height
    sq2 = numpy.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * threshold
    b3  = -2 * threshold * (height + width)
    c3  = (threshold - 1) * width * height
    sq3 = numpy.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)