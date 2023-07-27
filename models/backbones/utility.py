
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

from enum import Enum
from models.backbones.convolutions import Convolution

# the 'stackLayers' function returns a sequence of stacked layers of length 'modules', 
# each layer should have a constructor of format: 
#     layer(convSize, inputDimensions, outputDimensions, **args).
#
# the sequence will stack the layer with dimension changes applied in the first one:
# inputDimension -> outputDimension -> [...] -> outputDimension 

def stackLayers(convSize, inputDimension, outputDimension, modules, layer = Convolution, **kwargs):
    layers = [ layer(convSize, inputDimension, outputDimension, **kwargs) ]
    for i in range(1, modules):
        layers.append( layer (convSize, outputDimension, outputDimension, **kwargs) )
    return torch.nn.Sequential(*layers)

# unlike 'stackLayers', this function will stack the layers with dimension change in the last one:
# inputDimension -> [...] -> inputDimension -> outputDimension

def stackLayersReverted(k, inp_dim, out_dim, modules, layer = Convolution, **kwargs):
    layers = []
    for i in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return torch.nn.Sequential(*layers)

class MergeUp(torch.nn.Module):
    def forward(self, element1, element2):
        return element1 + element2

def mergeLayer(dim):
    return MergeUp()

def noLayer(dim):
    return None

def convolutionConv1x1(inputDimension, currentDimension, outputDimension):
    return torch.nn.Sequential(
        Convolution(3, inputDimension, currentDimension, batchNorm = False),
        torch.nn.Conv2d(currentDimension, outputDimension, (1, 1))
    )

def changeDimensionConv(inputDimension, outputDimension):
    return Convolution(3, inputDimension, outputDimension)

# indices is the top-K targets in a 2D tensor (batch, k), unsqueezed to (batch, k, 1) and
#     expanded the 3-rd dimension to (batch, k, dimension). the value of the k-th indices
#     vary between 0 and W*H. <extractTopK>
# and the 'feature' in the format of (batch, W*H, dimension).
# select the corresponding prediction in the format of (batch, k, dimension).

def gatherFeatures(feature, indices, mask = None):
    dimension  = feature.size(2)
    indices  = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), dimension)
    feature = feature.gather(1, indices)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feature)
        feature = feature[mask]
        feature = feature.view(-1, dimension)
    return feature

# when the kernelSize is set to 1, the algorithm is useless.
def nonMaximumSuppression(heat, kernelSize = 3):
    paddings = (kernelSize - 1) // 2
    
    hmax = F.max_pool2d(heat, (kernelSize, kernelSize), stride = 1, padding = paddings)
    keep = (hmax == heat).float()
    return heat * keep

def reshapeGatherFeatures(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gatherFeatures(feat, ind)
    return feat

# extract the top-k values from the heatmap in the format of ('batch', 'categories', H, W).
# the k values are originated in all the categories from one sample in the batch. thus
# a batch-length collection [{1, 2, ..., k}, {1, 2, ..., k} ...]

# this function returns a tuple of (values, index, category, y, x).
# each of them are a 2-D tensor (batch, K).
def extractTopK(scores, K = 20):

    batch, category, height, width = scores.size()
    topKScores, topKIndices = torch.topk(scores.view(batch, -1), K)

    # get the corresponding category number from index in the flattened score.
    # topKIndices : (batch, K) 2-D tensor
    topKCategories = (topKIndices / (height * width)).int()

    topKIndices = topKIndices % (height * width)
    topKY   = (topKIndices / width).int().float()
    topKX   = (topKIndices % width).int().float()
    return topKScores, topKIndices, topKCategories, topKY, topKX

def clampSigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

# a 3x3 basic convolution with padding.
def convolution3x3(inputDimension, outputDimension, stride = 1):
    return torch.nn.Conv2d(inputDimension, outputDimension, kernel_size = 3, stride=stride,
                           padding = 1, bias = False)

