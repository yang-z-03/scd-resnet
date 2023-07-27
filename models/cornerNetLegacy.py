
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

# python utilities ...

import random
import zipfile
import argparse
import pprint
from distutils.log import error, warn
from math import sqrt
from enum import Enum

from models.backbones.convolutions import Convolution
from models.backbones.pooling import poolingLayer, unpoolingLayer
from models.backbones.residuals import Residual, residual3x3
from models.backbones.hourglass import Hourglass
from models.losses.embeddings import embeddingLoss
from models.losses.focal import focalLoss
from models.losses.regression import smoothL1LossMask
from models.backbones.cornerPooling import TopPool, LeftPool, BottomPool, RightPool
from models.backbones.utility import \
    changeDimensionConv, clampSigmoid, convolutionConv1x1, extractTopK, gatherFeatures, mergeLayer, noLayer, \
    nonMaximumSuppression, reshapeGatherFeatures, stackLayers, stackLayersReverted

class Corner(Enum):
    TopLeft     = 0,
    TopRight    = 1,
    BottomLeft  = 2,
    BottomRight = 3

class StackHourglassForCornerNet(torch.nn.Module):

    # the CornerNet is constructed with an hourglass backbone and a prediction network.
    # the network architecture is demonstrated as follows:
    #
    #           [ preprocess (B, 3, H, W) ]
    #                       |
    #                       | Conv(7, 2) [3 -> 128]
    #           .preprocess | Residual(3, 2) [128 -> 'currentDimension']
    #                       v
    #           [ backbone input (B, 'currentDimension', H/4, W/4) ] -----------,
    #                       |                                                   |
    #                       | Hourglass ['currentDimension' ->                  |
    #                       |            'currentDimension']                    |
    #                       v                                                   |
    #           [ kp (B, 'currentDimension', H/4, W/4) ]                        |
    #                       |                                                   |
    #                       | Redim. Convolution ['currentDimension'            | Shortcut
    #                       | -> 'predictionDimension']                         | Layers
    #                       v                                                   | ['currentDimension' ->
    #           [ cnv (B, 'predictionDimension', H/4, W/4) ]                    |  'currentDimension']
    #                       |                                                   |
    #  heat <--- tlConv <---|                                                   |
    #   tag <--|            |                                                   |
    #  regr <--'            |                                                   |
    #   ... <--- brConv <---|                                                   |
    #  ['prediction' ->     |                                                   |
    #   'current']          | Convolution Previous Hourglass                    |
    #                       | ['predictionDimension' ->                         |
    #                       |  'currentDimension']                              |
    #                       v                                                   |
    #                       + <-------------------------------------------------'
    #                       |
    #                       | relu
    #                       v
    #            [ inter1 (B, 'currentDimension', H/4, W/4) ]
    #                       |
    #                       | Inter-Hourglass Layers
    #                       | ['currentDimension' -> 'currentDimension']
    #                       v 
    #            [ backbone input (B, 'currentDimension', H/4, W/4)]
    #                       |
    #                       ...

    # the output format of this network includes:
    # - two (N, 'outputdDimension', H, W) heatmap for top-left and bottom-right corner.
    # - two (N, 1, H, W) embedding index tags.
    # - two (N, 2, H, W) regression offsets, the 2 channels represent X and Y respectively.

    # for a more generalized version, see StackHourglass.py

    def __init__(
        self, hourglassIteration, hourglassStacks, dimensions, modules, outputDimension,
        beforeBackbone = None, predictionConvDim = 256, 

        makeTopLeft = noLayer, makeBottomRight = noLayer,
        makeConvolutionLayer = changeDimensionConv, makeHeatmapLayer = convolutionConv1x1,
        makeTagLayer = convolutionConv1x1, makeRegressionLayer = convolutionConv1x1,

        hourglassBeforeDownsample = stackLayers, hourglassCentral = stackLayers, 
        hourglassBefore = stackLayers, hourglassAfter = stackLayersReverted,
        hourglassPool = poolingLayer, hourglassUnpooling = unpoolingLayer,
        hourglassMerge = mergeLayer, makeInternalLayer = residual3x3, 
        hourglassLayer = Residual ):

        super(StackHourglassForCornerNet, self).__init__()

        self.hourglassStacks = hourglassStacks
        self.decoder = decodeCornerNet

        currentDimension = dimensions[0]

        self.preprocess = torch.nn.Sequential(
            Convolution ( 7, 3, 128, stride = 2 ),
            # Residual ( 3, 128, 256, stride = 2 )
            Residual ( 3, 128, currentDimension, stride = 2 ) # CHANGE
        ) if beforeBackbone is None else beforeBackbone

        self.hourglassStack  = torch.nn.ModuleList([
            Hourglass(
                # FIX: the dimensions have one element left?
                hourglassIteration, dimensions, modules, layer = hourglassLayer,
                layersBeforeDownsample = hourglassBeforeDownsample,
                layersCentral = hourglassCentral,
                layersBeforeHourglass = hourglassBefore,
                layersAfterHourglass = hourglassAfter,
                layersDownsampling = hourglassPool,
                layersUpsampling = hourglassUnpooling,
                layersMerge = hourglassMerge
            ) for _ in range(hourglassStacks)
        ])

        # the output size of the hourglass backbone has 'currentDimension', and this 
        # reshapes the output to 'predictionConvDim'.
        self.redimConvolution = torch.nn.ModuleList([
            makeConvolutionLayer(currentDimension, predictionConvDim) for _ in range(hourglassStacks)
        ])

        # the networks applied after redimension
        self.topLeftConvs = torch.nn.ModuleList([
            makeTopLeft(predictionConvDim) for _ in range(hourglassStacks)
        ])

        self.bottomRightConvs = torch.nn.ModuleList([
            makeBottomRight(predictionConvDim) for _ in range(hourglassStacks)
        ])

        # the network to predict heatmaps from 'predictionConvDim' -> 'outputDimension'.
        self.topLeftHeatmaps = torch.nn.ModuleList([
            makeHeatmapLayer(predictionConvDim, currentDimension, outputDimension) \
                for _ in range(hourglassStacks)
        ])

        self.bottomRightHeatmaps = torch.nn.ModuleList([
            makeHeatmapLayer(predictionConvDim, currentDimension, outputDimension) \
                for _ in range(hourglassStacks)
        ])

        # the network to predict the tags(embeddings id). this helps distinguish the vertex
        # either to top-left or to botton-right category, using associative embeddings
        # (A. Newell, J. Deng 2017. 2018.)

        self.topLeftTags  = torch.nn.ModuleList([
            makeTagLayer(predictionConvDim, currentDimension, 1) \
                for _ in range(hourglassStacks)
        ])

        self.bottomRightTags  = torch.nn.ModuleList([
            makeTagLayer(predictionConvDim, currentDimension, 1) \
                for _ in range(hourglassStacks)
        ])

        # magic initialization of bias?
        #
        #     return torch.nn.Sequential(
        #         Convolution(3, inputDimension, currentDimension, with_bn=False),
        # ++>     torch.nn.Conv2d(currentDimension, outputDimension, (1, 1))
        #     )
        #
        # <models/backbones/utility.py> convolutionConv1x1
        # bias of the pointed convolution, the 1x1.

        for tlHeat, brHeat in zip(self.topLeftHeatmaps, self.bottomRightHeatmaps):
            tlHeat[-1].bias.data.fill_(-2.19)
            brHeat[-1].bias.data.fill_(-2.19)

        # layers between the stacked hourglass.
        self.interHourglassLayers = torch.nn.ModuleList([
            makeInternalLayer(currentDimension) for _ in range(hourglassStacks - 1)
        ])

        self.shortcutLayers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(currentDimension, currentDimension, (1, 1), bias = False),
                torch.nn.BatchNorm2d(currentDimension)
            ) for _ in range(hourglassStacks - 1)
        ])

        self.convPrevHourglass = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(predictionConvDim, currentDimension, (1, 1), bias = False),
                torch.nn.BatchNorm2d(currentDimension)
            ) for _ in range(hourglassStacks - 1)
        ])

        self.topLeftRegressions = torch.nn.ModuleList([
            makeRegressionLayer(predictionConvDim, currentDimension, 2) for _ in range(hourglassStacks)
        ])

        self.bottomRightRegressions = torch.nn.ModuleList([
            makeRegressionLayer(predictionConvDim, currentDimension, 2) for _ in range(hourglassStacks)
        ])

        self.relu = torch.nn.ReLU(inplace = True)

    # the architecture is drawn above.
    def trainNetwork(self, *args):
        image   = args[0]
        tl_inds = args[1]
        br_inds = args[2]

        # prepare the image from an original resolution to those fitted with the 
        # hourglass backbone's input.
        inter = self.preprocess(image)
        outs  = []

        layers = zip(
            self.hourglassStack, self.redimConvolution,
            self.topLeftConvs, self.bottomRightConvs,
            self.topLeftHeatmaps, self.bottomRightHeatmaps,
            self.topLeftTags, self.bottomRightTags,
            self.topLeftRegressions, self.bottomRightRegressions
        )

        for id, layer in enumerate(layers):
            hourglass, redimConv = layer[0:2]
            tlConv, brConv = layer[2:4]
            tlHeat, brHeat = layer[4:6]
            tlTag, brTag = layer[6:8]
            tlRegression, brRegression = layer[8:10]

            kp  = hourglass(inter)
            cnv = redimConv(kp)

            tl_cnv = tlConv(cnv)
            br_cnv = brConv(cnv)

            tl_heat, br_heat = tlHeat(tl_cnv), brHeat(br_cnv)
            tl_tag,  br_tag  = tlTag(tl_cnv),  brTag(br_cnv)
            tl_regr, br_regr = tlRegression(tl_cnv), brRegression(br_cnv)

            # these output of reshapeGatherFeatures in the shape of (batch, k, dimension).
            tl_tag  = reshapeGatherFeatures(tl_tag, tl_inds)
            br_tag  = reshapeGatherFeatures(br_tag, br_inds)
            tl_regr = reshapeGatherFeatures(tl_regr, tl_inds)
            br_regr = reshapeGatherFeatures(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if id < self.hourglassStacks - 1:
                inter = self.shortcutLayers[id](inter) + self.convPrevHourglass[id](cnv)
                inter = self.relu(inter)
                inter = self.interHourglassLayers[id](inter)

        return outs

    def evalNetwork(self, *xs, **kwargs):
        image = xs[0]

        inter = self.preprocess(image)
        outs  = []

        layers = zip(
            self.hourglassStack, self.redimConvolution,
            self.topLeftConvs, self.bottomRightConvs,
            self.topLeftHeatmaps, self.bottomRightHeatmaps,
            self.topLeftTags, self.bottomRightTags,
            self.topLeftRegressions, self.bottomRightRegressions
        )

        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.hourglassStacks - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.hourglassStacks - 1:
                inter = self.shortcutLayers[ind](inter) + self.convPrevHourglass[ind](cnv)
                inter = self.relu(inter)
                inter = self.interHourglassLayers[ind](inter)

        return self.decoder(*outs[-6:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self.trainNetwork(*xs, **kwargs)
        return self.evalNetwork(*xs, **kwargs)

# uses the information in the top-left, and bottom-right heatmap to interpret locations.
# where the heatmap in the format of ('batch', 'dimensions', H, W) 4-D tensor.
# heatmaps has 'outputDimension' channels, tags have 1 channel, and regression has 2 channels.

# average embedding threshold: \Delta = 1.

def decodeCornerNet(
    topLeftHeatmap, bottomRightHeatmap, topLeftTags, bottomRightTags, 
    topLeftRegression, bottomRightRegression, 
    K = 100, nmsKernelSize = 1, avgEmbeddingThreshold = 1, detectionCount = 1000 ):

    batch, category, height, width = topLeftHeatmap.size()

    # apply sigmoid uniform to the heatmap, limiting its value to [0, 1].
    topLeftHeatmap = torch.sigmoid(topLeftHeatmap)
    bottomRightHeatmap = torch.sigmoid(bottomRightHeatmap)

    # perform Non-Maximum Suppression (NMS) on heatmaps
    topLeftHeatmap = nonMaximumSuppression(topLeftHeatmap, kernelSize = nmsKernelSize)
    bottomRightHeatmap = nonMaximumSuppression(bottomRightHeatmap, kernelSize = nmsKernelSize)

    # all of these are in format of (batch, k)
    tlScores, tlIndices, tlCategories, tlY, tlX = extractTopK(topLeftHeatmap, K=K)
    brScores, brIndices, brCategories, brY, brX = extractTopK(bottomRightHeatmap, K=K)

    # for the output x and ys in the shape of (batch, k). this expansion expands top-left
    # elements to the row and bottom right elements to the column. for example, in a given
    # batch, the x and y are [a, b, c]. (k)

    # top-left elements are expanded to 
    # [ [a, a, a]
    #   [b, b, b]
    #   [c, c, c] ] (k, k)
    # and bottom-right elements are expanded to 
    # [ [a, b, c]
    #   [a, b, c]
    #   [a, b, c] ] (k, k).

    tlY = tlY.view(batch, K, 1).expand(batch, K, K)
    tlX = tlX.view(batch, K, 1).expand(batch, K, K)
    brY = brY.view(batch, 1, K).expand(batch, K, K)
    brX = brX.view(batch, 1, K).expand(batch, K, K)

    if topLeftRegression is not None and bottomRightRegression is not None:
        
        # after the reshape feature gathering, the regression now have (batch, k, dim = 2)
        topLeftRegression = reshapeGatherFeatures(topLeftRegression, tlIndices)
        topLeftRegression = topLeftRegression.view(batch, K, 1, 2)
        bottomRightRegression = reshapeGatherFeatures(bottomRightRegression, brIndices)
        bottomRightRegression = bottomRightRegression.view(batch, 1, K, 2)

        # the addition of different-size tensor, tlX : (batch, k, k) and 
        # topLeftRegression[..., 0] : (batch, k, 1). 

        # (batch, k, k) vector with regression (offset) added to the corresponding coordinate.
        tlX = tlX + topLeftRegression[..., 0]
        tlY = tlY + topLeftRegression[..., 1]
        brX = brX + bottomRightRegression[..., 0]
        brY = brY + bottomRightRegression[..., 1]

    # all possible boxes based on top k corners (ignoring class) (batch, k, k, 4).
    bboxes = torch.stack((tlX, tlY, brX, brY), dim = 3)

    topLeftTags = reshapeGatherFeatures(topLeftTags, tlIndices)
    topLeftTags = topLeftTags.view(batch, K, 1)
    bottomRightTags = reshapeGatherFeatures(bottomRightTags, brIndices)
    bottomRightTags = bottomRightTags.view(batch, 1, K)

    # for top-k ids in one batch
    # topLeft: [[ [ id_{t1} ], 
    #             [ id_{t2} ],
    #             [ id_{tk} ] ]]
    #
    # bottomRight: [[ [ id_{b1}, id_{b2}, id_{b3} ] ]].
    #
    # dists: [[ [ t1 - b1, t1 - b2, t1 - bk ],
    #           [ t2 - b1, t2 - b2, t2 - bk ],
    #           [ tk - b1, tk - b2, tk - bk ] ]].
    
    embeddingIndexDistance  = torch.abs(topLeftTags - bottomRightTags)

    # in fact, the expand operation can be omitted?
    tlScores = tlScores.view(batch, K, 1).expand(batch, K, K)
    brScores = brScores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tlScores + brScores) / 2

    # reject boxes based on classes
    tlCategories = tlCategories.view(batch, K, 1).expand(batch, K, K)
    brCategories = brCategories.view(batch, 1, K).expand(batch, K, K)
    idCategoryNotMatch = (tlCategories != brCategories)

    # reject boxes based on index distance
    idEmbeddingIndexNotMatch = (embeddingIndexDistance > avgEmbeddingThreshold)

    # reject boxes based on widths and heights: that the bottom-right corner must at
    # the downside and rightside of the top-left corner
    idNotRight  = (brX < tlX)
    idNotBottom = (brY < tlY)

    scores[idCategoryNotMatch] = -1
    scores[idEmbeddingIndexNotMatch] = -1
    scores[idNotRight] = -1
    scores[idNotBottom] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, detectionCount)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = gatherFeatures(bboxes, inds)

    categories = tlCategories.contiguous().view(batch, -1, 1)
    categories = gatherFeatures(categories, inds).float()

    tlScores = tlScores.contiguous().view(batch, -1, 1)
    tlScores = gatherFeatures(tlScores, inds).float()
    brScores = brScores.contiguous().view(batch, -1, 1)
    brScores = gatherFeatures(brScores, inds).float()

    detections = torch.cat([bboxes, scores, tlScores, brScores, categories], dim=2)
    return detections

# the corner pool gathers the information from graphic features to infer the location
# of bounding points. a line/column maximum pool.
#
# after the prediction network redimension, apply this corner pooling layers.
#
#                             [(N, 'predictionDim', H, W)]
#                                          |
#                 ,------------------------+------------------------,
#         branch1 |                branch2 |                        |
#                 v                        v                        | shortcut conv
#         [(N, 128, H, W)]          [(N, 128, H, W)]                v
#                 |                        |           [(N, 'predictionDim', H, W)]
# branch pooling1 |       branch pooling 2 |                        |
#                 v                        v                        | bn
#         [(N, 128, H, W)]          [(N, 128, H, W)]                |
#                 |                        |                        |
#                 '----------> + <---------'                        |
#                              |                                    |
#                              | branch merge                       |
#                              v                                    |
#                 [(N, 'predictionDim', H, W)]                      |
#                              |                                    |
#                              | branch merge bn                    |
#                              '----------------> + <---------------'
#                                                 |
#                                                 | ReLU
#                                                 | lastConv
#                                                 v
#                                    [(N, 'predictionDim', H, W)]

class CornerPool(torch.nn.Module):

    def __init__(self, predictionDimension, pool1, pool2):
        super(CornerPool, self).__init__()

        self.branch1 = Convolution(3, predictionDimension, 128)
        self.branch2 = Convolution(3, predictionDimension, 128)

        self.branchMerge = torch.nn.Conv2d(128, predictionDimension, (3, 3), padding=(1, 1), bias=False)
        self.branchMergeBn = torch.nn.BatchNorm2d(predictionDimension)

        self.shortcutConv = torch.nn.Conv2d(predictionDimension, predictionDimension, (1, 1), bias = False)
        self.shortcutBn = torch.nn.BatchNorm2d(predictionDimension)
        self.mixReLU = torch.nn.ReLU(inplace = True)

        self.lastConv = Convolution(3, predictionDimension, predictionDimension)

        self.branchPooling1 = pool1()
        self.branchPooling2 = pool2()

    def forward(self, x):
        
        # pool 1
        p1_conv1 = self.branch1(x)
        pool1    = self.branchPooling1(p1_conv1)

        # pool 2
        p2_conv1 = self.branch2(x)
        pool2    = self.branchPooling2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.branchMerge(pool1 + pool2)
        p_bn1   = self.branchMergeBn(p_conv1)

        conv1 = self.shortcutConv(x)
        bn1   = self.shortcutBn(conv1)
        relu1 = self.mixReLU(p_bn1 + bn1)

        conv2 = self.lastConv(relu1)
        return conv2

class TopLeftPool(CornerPool):
    def __init__(self, dim):
        super(TopLeftPool, self).__init__(dim, TopPool, LeftPool)

class BottomRightPool(CornerPool):
    def __init__(self, dim):
        super(BottomRightPool, self).__init__(dim, BottomPool, RightPool)

def makeTopLeftLayer(dim):
    return TopLeftPool(dim)

def makeBottomRightLayer(dim):
    return BottomRightPool(dim)

def makePoolLayer(dim):
    return torch.nn.Sequential()

def makeHourglassLayer(kernelSize, inputDimension, outputDimension, modules, layer = Convolution, **kwargs):
    layers  = [layer(kernelSize, inputDimension, outputDimension, stride = 2)]
    layers += [layer(kernelSize, outputDimension, outputDimension) for _ in range(modules - 1)]
    return torch.nn.Sequential(*layers)

class CornerNet(StackHourglassForCornerNet):
    def __init__(self, **kwargs):

        hourglassIters     = 5
        dimensions         = [256, 256, 384, 384, 384, 512]
        modules            = [2, 2, 2, 2, 2, 4]
        outputDimensions   = 80

        super(CornerNet, self).__init__(
            hourglassIters, 2, dimensions, modules, outputDimensions,
            makeTopLeft = makeTopLeftLayer,
            makeBottomRight = makeBottomRightLayer,
            hourglassPool = makePoolLayer,
            hourglassBefore = makeHourglassLayer,
            hourglassLayer = Residual, predictionConvDim = 256
        )

class CornerNetLoss(torch.nn.Module):

    def __init__( self, pullWeight = 1, pushWeight = 1, regressionWeight = 1, 
        focal = focalLoss):

        super(CornerNetLoss, self).__init__()

        self.pullWeight = pullWeight
        self.pushWeight = pushWeight
        self.regressionWeight = regressionWeight
        self.focal = focal
        self.embedding = embeddingLoss
        self.regression = smoothL1LossMask

    # evaluate loss between the output of the network and targets.
    # several points worth noticing in this loss application:
    # - multiple midterm training in after hourglass module. that after every hourglass
    #   in a stack, the prediction modules produces an result and apply the loss.
    # - clamped sigmoid to avoid NaNs and Infs.

    # the input format are the direct network output.
    # (..., tlHeat Cx, brHeat Cx, tlEmbedding 1x, brEmbedding 1x, tlOffset 2x, brOffset 2x).

    def forward(self, outs, targets):
        stride = 6

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_tags  = outs[2::stride]
        br_tags  = outs[3::stride]
        tl_regrs = outs[4::stride]
        br_regrs = outs[5::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss of heatmap
        focalL = 0

        tl_heats = [clampSigmoid(t) for t in tl_heats]
        br_heats = [clampSigmoid(b) for b in br_heats]

        focalL += self.focal(tl_heats, gt_tl_heat)
        focalL += self.focal(br_heats, gt_br_heat)

        # the pull and push loss of the embedding tag
        pullL = 0
        pushL = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.embedding(tl_tag, br_tag, gt_mask)
            pullL += pull
            pushL += push
        pullL = self.pullWeight * pullL
        pushL = self.pushWeight * pushL

        offsetL = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            offsetL += self.regression(tl_regr, gt_tl_regr, gt_mask)
            offsetL += self.regression(br_regr, gt_br_regr, gt_mask)
        offsetL = self.regressionWeight * offsetL

        loss = (focalL + pullL + pushL + offsetL) / len(tl_heats)

        # the loss is one real number, this give a dimension to be a one-element array.
        # torch.tensor([loss]).
        return loss.unsqueeze(0)
