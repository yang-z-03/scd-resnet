
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

# the embedding tag input in the format of (N, k, 1).
# mask indicates the area that is required to assign embedding tags. in CornerNet, the whole
# image is probable for the existence of a corner, so that the mask is an all-one matrix.
# every pair of corner (top-left, bottom-right) shares one unique embedding tag.

# the mask is in the format of (N, max-tag-len = 128).

def embeddingLoss(tagTopLeft, tagBottomRight, mask):

    numberOfObjs = mask.sum( dim = 1, keepdim = True ).float()
    
    tagTopLeft = tagTopLeft.squeeze()
    tagBottomRight = tagBottomRight.squeeze() # (N, k)

    tagMean = (tagTopLeft + tagBottomRight) / 2

    tagTopLeft = torch.pow(tagTopLeft - tagMean, 2) / (numberOfObjs + 1e-4)
    tagTopLeft = tagTopLeft[mask].sum()
    tagBottomRight = torch.pow(tagBottomRight - tagMean, 2) / (numberOfObjs + 1e-4)
    tagBottomRight = tagBottomRight[mask].sum()

    # the pull loss, smaller if the two embedding tags from one boundary box is 
    # close to each other.
    pull = tagTopLeft + tagBottomRight

    # a mask of one picture may be [1, 1, 0, 0, 0, 0, 0, 0, ...] indicating there are
    # two objects on it for detection. this operation yields
    #
    # [[2, 2, 1, 1, ...],
    #  [2, 2, 1, 1, ...],
    #  [1, 1, 0, 0, ...],
    #  [1, 1, 0, 0, ...],
    #  ...              ]
    #
    # thus mask selects the N^2 region (that contains 2) in max-tag-len^2 space.

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2) 
    numberOfObjs  = numberOfObjs.unsqueeze(2)

    NxNminus1 = (numberOfObjs - 1) * numberOfObjs # N(N-1)
    dist = tagMean.unsqueeze(1) - tagMean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = F.relu(dist, inplace = True)
    # the sum of matrix calculate the actual sum N times
    dist = dist - 1 / (numberOfObjs + 1e-4) 
    dist = dist / (NxNminus1 + 1e-4)
    dist = dist[mask]

    # the push loss, greater if average embeddings from two boundary boxes is close 
    # to each other
    push = dist.sum()
    return pull, push
