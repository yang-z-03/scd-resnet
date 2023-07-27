
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

def focalLoss(prediction, groundTruth, alpha = 2, beta = 4):

    positiveIndices = groundTruth.eq(1)
    negativeIndices = groundTruth.lt(1)

    negativeWeights = torch.pow(1 - groundTruth[negativeIndices], beta)

    loss = 0

    # the first dimension (samples in a batch) is splitted in this iteration.
    for pred in prediction:
        positivePrediction = pred[positiveIndices]
        negativePrediction = pred[negativeIndices]

        positiveLoss = torch.log(positivePrediction) * torch.pow(1 - positivePrediction, alpha)
        negativeLoss = torch.log(1 - negativePrediction) * torch.pow(negativePrediction, alpha) * negativeWeights

        positiveCount  = positiveIndices.float().sum()
        positiveLoss = positiveLoss.sum()
        negativeLoss = negativeLoss.sum()

        # that the image has no positive point, all the points are negative targets.
        if positivePrediction.nelement() == 0:
            loss = loss - negativeLoss

        else:
            loss = loss - (positiveLoss + negativeLoss) / positiveCount

    return loss