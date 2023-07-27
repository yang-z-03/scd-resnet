
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

# regression and ground-truth in the format of (N, k, 2),
# mask in the format of (N, max-tag-len = 128).

def smoothL1LossMask(regression, groundTruth, mask):
    num  = mask.float().sum()
    mask = mask.bool()
    mask = mask.unsqueeze(2).expand_as(groundTruth)
    
    regr_loss = F.smooth_l1_loss(regression[mask], groundTruth[mask], reduction = 'sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def L1LossMask(regression, groundTruth, mask):
    num  = mask.float().sum()
    mask = mask.bool()
    mask = mask.unsqueeze(2).expand_as(groundTruth)
    
    regr_loss = F.l1_loss(regression[mask], groundTruth[mask], reduction = 'sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss