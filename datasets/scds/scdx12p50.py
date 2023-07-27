
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
from random import shuffle
import zipfile
import json
from tqdm import tqdm
import math
import shutil

from datasets.argumentations import PaddingMode, ResampleMode, gaussianNoise, normalize, \
    randomRotate, rotate, varianceJitter
from evaluations.intersection import centerThresholdRadius
from datasets.utility import gaussianMargin2D
from configuration import defaultConfig
from logger import Logger
import matplotlib.pyplot as plt

torch.random.manual_seed(42)
numpy.random.seed(42)

MAXTAGLEN = 30

TARGETSIZE = 512
TARGETSIZEH = 256
HEATMAPSIZE = 128
DOWNSAMPLE = int(TARGETSIZE/HEATMAPSIZE)
THRESHOLDIOU = 0.5

TESTSET = 5760
REALTIMETEST = 5760

ARGUMENTRATIO = 12
PARTITION = 0.50

TRAINSUBSET = 'train12p50'

class SCD(Dataset):

    def __init__(self, zipPath, useGPU, dataSplit = None):

        # initialize the confocal dataset from a preprocesser-generated zip file. this
        # zip file has the following subdirectories.
        # 
        # ./
        #     object-count.json
        #     dataset.json
        # ./samples
        #     1.1.npy
        # ./locs
        #     1.1.npy
        #
        # the {a}.{b}.npy is the binary stored numpy matrix in the format of (H, W).
        # which stored the generated heatmaps in 'tl' and 'br', the original grayscale image
        # in 'samples', and the rectangle boundaru in the format of (K, 4) in 'locs' 
        # for [ctX, ctY, W, H].
        # {a} indicates the index of original WSI. and {b} is the clip number.
        # 
        # object-count.json at the root directory is a json file with format
        # 
        # { "count" : {
        #         "101.1" : 3, ...
        #     }
        # }
        #
        # names.json contains the array of names in json 

        # check if the dataset is already extracted.

        # tempDir = defaultConfig.dirTemp + defaultConfig.datasetName + "/"
        tempDir = defaultConfig.dirTemp + 'confocalCenter' + "/"
        self.extracted = True if os.path.exists(tempDir) else False
            
        if self.extracted == False:
            archieve = zipfile.ZipFile(zipPath)
            Logger.log("Extracting Data to Temporary Directory ...")
            archieve.extractall(tempDir)
        
        names = []

        self.samples = []
        self.bounds = []
        self.count = 0
        self.names = names
        self.objectCounts = {}

        with open(tempDir + "dataset.json", "r") as dsfile:
            content = dsfile.read()
            datasetJson = json.loads(content)
            self.names = datasetJson["names"]
            self.count = len(self.names)
        
        with open(tempDir + "object-count.json", "r") as ocfile:
            content = ocfile.read()
            self.objectCounts = json.loads(content)
        
        # os.remove(tempDir + "dataset.json")
        # os.remove(tempDir + "object-count.json")

        progress = tqdm(self.names, ncols = 100)
        for name in progress:
            
            progress.set_description("Loading samples - {: >8}".format(name))

            sample = numpy.load(tempDir + "samples/{}.npy".format(name))
            locs = numpy.load(tempDir + "locs/{}.npy".format(name))

            # uniform the tensor to (1, H, W).
            self.samples.append(torch.from_numpy(sample).unsqueeze(0).float())
            self.bounds.append(torch.from_numpy(locs).float())
        
        # shuffle the samples and pick out 10 percent of images for validation.

        # to accomplish tests on the saturation of training samples and random rotations, we need to carefully
        # select counting orders and keep test set identical.

        # pick a number of full-slide images from the dataset to simulate a declined dataset. we choose in the
        # unit of subset images at random.

        FSI = 130
        ARGUM = 16
        CLIP = 24

        rawIndex = 0
        self.order = []
        for fsi in range(FSI):
            for argum in range(ARGUM):
                for clip in range(CLIP):

                    # apply the intake standards for images.
                    if argum < ARGUMENTRATIO:
                        self.order += [rawIndex]
                    rawIndex += 1
        
        # picks the corresponding partiton of the entire dataset
        shuffle(self.order)
        self.order = self.order[0: int(len(self.order) * PARTITION)]
                    
        self.dataProfile = {'validation': []}
        
        if dataSplit is None:
            Logger.log("The Data Split Profile Do Not Exist, We Randomly Select 10 pct. of Samples as Validation Set.")
            shuffle(self.order)
            numValidation = round(TESTSET)
            self.dataProfile['validation'] = self.order[0:numValidation]
            self.order = self.order[numValidation:]
            self.count = len(self.order)
            self.dataProfile[TRAINSUBSET] = self.order
            
        else:
            Logger.log("Extracting Validation Set from Data Split Profile ...")
            self.dataProfile = dataSplit
            
            if TRAINSUBSET in self.dataProfile.keys():
                self.order = self.dataProfile[TRAINSUBSET]
            else:
                for x in range(self.count):
                    if x in self.dataProfile['validation'] and x in self.order:
                        self.order.remove(x)
                self.dataProfile[TRAINSUBSET] = self.order
            
            self.count = len(self.order)
        
        Logger.log("Building Validation Set ...")

        self.validSamples = []
        self.validHeat    = []
        self.validRegr    = []
        self.validTagMask = []
        self.validLocs    = []
        self.validObjNum  = []
        self.validInds    = []

        totalRealtimeValid = 0
        for id in self.dataProfile['validation']:
            totalRealtimeValid += 1
            
            if totalRealtimeValid > REALTIMETEST:
                continue
            
            sample, locs = ( self.samples[id], self.bounds[id] )
            heat = torch.zeros(HEATMAPSIZE, HEATMAPSIZE)
            
            for loc in locs:
                loc[0] = int(loc[0])
                loc[1] = int(loc[1])
                radius = centerThresholdRadius(
                    2 * math.sqrt(loc[4] ** 2 + loc[5] ** 2),
                    2 * loc[6].cpu().item(),
                    THRESHOLDIOU
                )

                SCD.drawGaussian( (loc[0], loc[1]), heat, radius)

            heat = heat.unsqueeze(0)

            sample = normalize(sample)
            heatIndices = torch.zeros(MAXTAGLEN)
            tagMask = torch.zeros(MAXTAGLEN)
            tagMask[:len(locs)] = 1

            if len(locs) > 0:
                heatIndices[:len(locs)] = torch.floor(locs[:, 1]) * HEATMAPSIZE + torch.floor(locs[:, 0])
            regr = [[loc[2], loc[3], loc[4], loc[5], loc[6], loc[7]] for loc in locs]
            regr = torch.tensor(regr)
            
            if useGPU:
                sample = sample.cuda( non_blocking = True)
                heat = heat.cuda( non_blocking = True)
            
            self.validSamples += [sample]
            self.validHeat += [heat]
            self.validObjNum += [len(locs)]

            if len(regr) == 0:
                fixedRegr = torch.zeros(MAXTAGLEN, 6)
            else:
                fixedRegr = torch.zeros(MAXTAGLEN, 6)
                fixedRegr[0:len(regr),:] = regr

            if len(locs) == 0:
                fixedLocs = torch.zeros(MAXTAGLEN, 8)
            else:
                fixedLocs = torch.zeros(MAXTAGLEN, 8)
                fixedLocs[0:len(locs),:] = locs
            
            if useGPU:
                fixedRegr = fixedRegr.cuda( non_blocking = True)
                tagMask = tagMask.cuda( non_blocking = True)
                fixedLocs = fixedLocs.cuda( non_blocking = True)
                heatIndices = heatIndices.cuda( non_blocking = True)
            
            '''
            # assert the corresponding locations on heatmaps
            targetMask = heat == 1
            if torch.sum(targetMask) != torch.sum(tagMask):
                Logger.err("sum of target heatmap peaks not match the object count. in {}".format(id))
                continue
            
            flag = 0
            targetMask = targetMask.reshape(HEATMAPSIZE * HEATMAPSIZE)
            for i in range(int(torch.sum(tagMask).item())):
                flag += targetMask[heatIndices[i].long()].byte()
            if flag != torch.sum(targetMask):
                Logger.err("target heatmap peaks are not in its correct positions. in {}".format(id))
                continue
            '''
            
            self.validRegr += [fixedRegr]
            self.validTagMask += [tagMask.bool()]
            self.validLocs += [fixedLocs]
            self.validInds += [heatIndices.long()]

        self.validation = {
            "xs": [ torch.stack(self.validSamples, 0),
                    torch.stack(self.validInds, 0) ],
            "ys": [ torch.stack(self.validHeat, 0),
                    torch.stack(self.validTagMask, 0),
                    torch.stack(self.validRegr, 0),
                    torch.stack(self.validLocs, 0),
                    self.validObjNum ]
        }
        
        validProfile = open(defaultConfig.dirDataSplitProfile, "w+")
        validProfile.write(json.dumps(self.dataProfile))
        validProfile.close()

        Logger.log("Building Validation Set Completely with {} Samples".format(len(self.validSamples)))

        self.useGPU = useGPU

        # Logger.log("Removing Data ...")
        # shutil.rmtree(tempDir + "samples/")
        # shutil.rmtree(tempDir + "heatmaps/")
        # shutil.rmtree(tempDir + "locs/")
        
    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if(index == 0):
            
            shuffle(self.order)
            # Logger.info(":: confocalCenter.py :: Reshuffle Indices")

            pass

        sample = self.samples[self.order[index]]
        locs = self.bounds[self.order[index]]

        if self.useGPU:
            sample = sample.to(torch.device("cuda:0"))
            locs = locs.to(torch.device("cuda:0"))
        
        sample, heat, locs = SCD.argumentation( sample, locs )

        if self.useGPU:
            heat = heat.to(torch.device("cuda:0"))

        # round locs into integral numbers ranging from [0, TARGETSIZE].
        # it is possible that the bounding boxes goes out of range after the data argumentation
        # and its random rotation. this leads to indices out of range.
        
        heatIndices = torch.zeros(MAXTAGLEN).long()
        tagMask = torch.zeros(MAXTAGLEN).bool()
        tagMask[:len(locs)] = 1

        for x in range(len(locs)):
            if (locs[x, 0] < 0 or locs[x, 0] >= HEATMAPSIZE) or \
               (locs[x, 1] < 0 or locs[x, 1] >= HEATMAPSIZE):
                tagMask[x] = 0

        if len(locs) > 0:
            heatIndices[:len(locs)] = torch.floor(locs[:, 1]) * HEATMAPSIZE + torch.floor(locs[:, 0])
        
        # heatIndices may be out of bound due to the random rotation argumentation, these indices
        # have a False tag mask, but requires to be valid index in later operation, we put a dummy
        # value 0 to the index with False tag mask.

        heatIndices[tagMask == 0] = 0

        fixedRegr = torch.zeros(MAXTAGLEN, 6)
        regr = [[loc[2], loc[3], loc[4], loc[5], loc[6], loc[7]] for loc in locs]
        regr = torch.tensor(regr)
        
        if len(regr) != 0:
            fixedRegr[0:len(regr),:] = regr
        
        if self.useGPU:
            heatIndices = heatIndices.to(torch.device("cuda:0"))
            tagMask = tagMask.to(torch.device("cuda:0"))
            fixedRegr = fixedRegr.to(torch.device("cuda:0"))
        
        sample = sample.squeeze(0)

        # plt.imshow(heat.cpu().reshape(128, 128).detach().numpy())
        # plt.show()

        # the destination data format:
        # {
        #     "xs": {
        #               [0] : (BATCHSIZE, 1, TARGETSIZE, TARGETSIZE)
        #           }
        #     "ys": {
        #               [0] : (BATCHSIZE, 1, TARGETSIZE, TARGETSIZE)
        #               [1] : (BATCHSIZE, MAXTAGLEN)
        #               [2] : (BATCHSIZE, MAXTAGLEN, 7)
        #               [3] : (BATCHSIZE, MAXTAGLEN)
        #           }
        # }

        return {
            "xs" : [sample],
            "ys" : [heat, tagMask, fixedRegr, heatIndices]
        }
    
    def getValidationSet(self):

        length = REALTIMETEST
        # should split the validation set if it is too large.

        splitSet = []
        if length > defaultConfig.validationBatchSize:
            for k in range(length // defaultConfig.validationBatchSize):
                size = defaultConfig.validationBatchSize
                current = {
                    'xs': [ self.validation['xs'][0][int(k * size):int((k + 1) * size)] ],
                    'ys': [ self.validation['ys'][0][int(k * size):int((k + 1) * size)],
                            self.validation['ys'][1][int(k * size):int((k + 1) * size)],
                            self.validation['ys'][2][int(k * size):int((k + 1) * size)],
                            self.validation['ys'][3][int(k * size):int((k + 1) * size)],
                            self.validation['ys'][4][int(k * size):int((k + 1) * size)],
                            self.validation['xs'][1][int(k * size):int((k + 1) * size)] ]
                }
                
                splitSet += [current]
            
            return splitSet
        
        total = {
                    'xs': [ self.validation['xs'][0] ],  # sample
                    'ys': [ self.validation['ys'][0],    # heatmap
                            self.validation['ys'][1],    # tagmask
                            self.validation['ys'][2],    # regression
                            self.validation['ys'][3],    # locs
                            self.validation['ys'][4],    # object numbers (deprecated)
                            self.validation['xs'][1] ]   # target indices
                }
        
        return [total]

    @staticmethod
    @torch.no_grad()
    def argumentation(sample, locs, noiseSV = 0.05, jitterSV = 0.05):

        objs = locs.shape[0]
        
        # sample in the format of (1, S, S) and locs in the format of (N, 4).
        # random flip
        if numpy.random.uniform() > 0.5:
            sample = torch.flip(sample, [2])
            if objs > 0:
                locs[:, 0] = HEATMAPSIZE - 1 - locs[:, 0]
                locs[:, 2] = -locs[:, 2]   # offset x
                locs[:, 4] = -locs[:, 4]   # major axis x

        if numpy.random.uniform() > 0.5:
            sample = torch.flip(sample, [1])
            if objs > 0:
                locs[:, 1] = HEATMAPSIZE - 1 - locs[:, 1]
                locs[:, 3] = -locs[:, 3]   # offset y
                locs[:, 5] = -locs[:, 5]   # major axis y
        
        sample = normalize(sample)
        sample = varianceJitter(sample, jitterSV)
        sample = gaussianNoise(sample, noiseSV)

        # change dimension to (1, 1, S, S) and rotate image.
        sample = sample.unsqueeze(0)
        
        '''
        # sbef = sample.reshape(512,512).cpu().numpy()

        # random rotate
        # angle = numpy.random.uniform() * 4 - 2
        angle = 0
        sample = rotate(sample,  angle, PaddingMode.MirrorPadding, ResampleMode.Bilinear)
        
        # saft = sample.reshape(512,512).cpu().numpy()

        # the random rotation operation induces the replication of objects.
        # any location set in locs will be replicated 9 times.
        repLocs = []

        for loc in locs:
            ctx, cty, offx, offy, majx, majy, minl, halo = loc
            repLocs += [[ctx, cty, offx, offy, majx, majy, minl, halo]]
            repLocs += [[ctx, -1-cty, offx, -offy, majx, -majy, minl, halo]]
            repLocs += [[ctx, 2*HEATMAPSIZE-1-cty, offx, -offy, majx, -majy, minl, halo]]

            repLocs += [[2*HEATMAPSIZE-1-ctx, cty, -offx, offy, -majx, majy, minl, halo]]
            repLocs += [[2*HEATMAPSIZE-1-ctx, -1-cty, -offx, -offy, -majx, -majy, minl, halo]]
            repLocs += [[2*HEATMAPSIZE-1-ctx, 2*HEATMAPSIZE-1-cty, -offx, -offy, -majx, -majy, minl, halo]]

            repLocs += [[-1-ctx, cty, -offx, offy, -majx, majy, minl, halo]]
            repLocs += [[-1-ctx, -1-cty, -offx, -offy, -majx, -majy, minl, halo]]
            repLocs += [[-1-ctx, 2*HEATMAPSIZE-1-cty, -offx, -offy, -majx, -majy, minl, halo]]
        
        repLocs = torch.tensor(repLocs)

        if objs > 0:
            repLocs = ConfocalCenter.rotateCoordinates(repLocs, HEATMAPSIZE // 2, angle)
        
        locs = []
        for reploc in repLocs:
            ctx, cty = int(reploc[0]), int(reploc[1])
            if (ctx >= 0 and ctx < HEATMAPSIZE and cty >= 0 and cty < HEATMAPSIZE):
                a,b,c,d,e,f,g,h = reploc # intentionally, to avoid bug
                locs += [[a,b,c,d,e,f,g,h]]
        
        locs = torch.tensor(locs).cuda()
        '''

        '''
        angle = numpy.random.uniform() * 90
        sample = rotate(sample,  angle, PaddingMode.ConstantPadding, ResampleMode.Bilinear, torch.mean(sample).cpu().item())
        
        repLocs = []

        # constant padding donot introduces extra targets.
        for loc in locs:
            ctx, cty, offx, offy, majx, majy, minl, halo = loc
            repLocs += [[ctx, cty, offx, offy, majx, majy, minl, halo]]
        
        repLocs = torch.tensor(repLocs)

        if objs > 0:
            repLocs = ConfocalCenter.rotateCoordinates(repLocs, HEATMAPSIZE // 2, angle)
        
        locs = []
        for reploc in repLocs:
            ctx, cty = int(reploc[0]), int(reploc[1])
            if (ctx >= 0 and ctx < HEATMAPSIZE and cty >= 0 and cty < HEATMAPSIZE):
                a,b,c,d,e,f,g,h = reploc # intentionally, to avoid bug
                locs += [[a,b,c,d,e,f,g,h]]
        
        locs = torch.tensor(locs).cuda()
        '''
        
        heat = torch.zeros(HEATMAPSIZE, HEATMAPSIZE)
            
        for loc in locs:
            loc[0] = int(loc[0])
            loc[1] = int(loc[1])

            if (loc[0] < 0 or loc[0] >= HEATMAPSIZE) or \
               (loc[1] < 0 or loc[1] >= HEATMAPSIZE) : continue

            radius = centerThresholdRadius(
                2 * math.sqrt(loc[4] ** 2 + loc[5] ** 2),
                2 * loc[6].cpu().item(),
                THRESHOLDIOU
            )

            SCD.drawGaussian( (loc[0], loc[1]), heat, radius)

        heat = heat.unsqueeze(0)
        
        if objs == 0:
            locs = locs.reshape(0, 9)

        return sample, heat, locs

    @staticmethod
    def rotateCoordinates(locs, targetSize, angle):
        
        # calculate the center x and ys for central rotation. clockwise.
        locs[:, 0:2] += ( 0.5 - targetSize )
        sinA = math.sin(- angle * math.pi / 180.0) # clockwise
        cosA = math.cos(- angle * math.pi / 180.0)
        distance = torch.sqrt( torch.pow(locs[:,0],2) + torch.pow(locs[:,1],2) )

        sin = locs[:,1] / distance
        cos = locs[:,0] / distance
        rotsin = sin * cosA + cos * sinA
        rotcos = cos * cosA - sin * sinA
        locs[:,1] = distance * rotsin
        locs[:,0] = distance * rotcos
        locs[:, 0:2] -= ( 0.5 - targetSize )

        # rotate vectors clockwise.
        # offsets
        modO = torch.sqrt( torch.pow(locs[:,2], 2) + torch.pow(locs[:,3], 2))
        modMask = modO == 0
        sinO = locs[:, 3] / modO
        cosO = locs[:, 2] / modO
        locs[:, 3] = modO * (sinO * cosA + cosO * sinA)
        locs[:, 2] = modO * (cosO * cosA - sinO * sinA)
        locs[:, 3][modMask] = 0
        locs[:, 2][modMask] = 0

        # major axis
        modMAJ = torch.sqrt( torch.pow(locs[:,4], 2) + torch.pow(locs[:,5], 2))
        sinMAJ = locs[:, 5] / modMAJ
        cosMAJ = locs[:, 4] / modMAJ
        locs[:, 5] = modMAJ * (sinMAJ * cosA + cosMAJ * sinA)
        locs[:, 4] = modMAJ * (cosMAJ * cosA - sinMAJ * sinA)

        return locs

    @staticmethod
    def drawGaussian(point, heatmap, radius):
        roi = math.ceil( radius * 2 )
        top, left, bottom, right = roi, roi, roi, roi
        x, y = point
        x = int(x.item()); y = int(y.item()) 
        
        height, width = heatmap.shape
        if x - left < 0: left = x
        if x + right >= width: right = width - x - 1
        if y - top < 0: top = y
        if y + bottom >= height: bottom = height - y - 1

        gauss = torch.tensor( gaussianMargin2D((left, top, right, bottom), radius / 3) )
        heatmap[ y - top: y + bottom + 1, x - left: x + right + 1] = \
            gauss + heatmap[ y - top: y + bottom + 1, x - left: x + right + 1]
        heatmap[heatmap > 1] = 1