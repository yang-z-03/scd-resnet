
# pytorch references ...
# and pytorch multi-gpu support

import math
import re
from matplotlib.transforms import Bbox
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
from math import ceil, sqrt, exp
from datasets.argumentations import PaddingMode, ResampleMode, rotate
from evaluations.intersection import centerThresholdRadius
from logger import Logger
from tqdm import tqdm
import json
import sys

torch.random.manual_seed(42)
numpy.random.seed(42)

# returns a H x W single-channel standardized image with a 0 mean and 1 stdvar.
def grayscale(path) -> numpy.ndarray:
    image = Image.open(path)

    # convert into grayscale image.
    colorImage = numpy.array(image)
    r = colorImage[:,:,0]
    g = colorImage[:,:,1]
    b = colorImage[:,:,2]
    numpyImage = 0.30 * r + 0.59 * g + 0.11 * b

    return numpyImage

def decode(path, imageName, width, height, iouThreshold = 0.7):
    attemptPath = path + os.path.splitext(imageName)[0] + ".txt"
    if (os.path.exists(attemptPath) == False):
        return None

    locations = []
    with open(path + os.path.splitext(imageName)[0] + ".txt") as annotation:
        content = annotation.readlines()
        heatmap = torch.zeros(height, width)

        for line in content:
            
            if len(line) <= 5 : continue

            # the positions are encoded in the order 
            # head.x; head.y; tail.x; tail.y; horizontal size; halo radius
            
            # and we convert this encode into the target information output in our network:
            # center (x, y)             - the integral center point of the target sperm in the /4x heatmap.
            # center offset (ox, oy)    - the offset correction of the center.
            # major axis vector (x, y)  - the major axis of the sperm, in /4x.
            # minor axis vector (x, y)  - the minor axis of the sperm, in /4x.
            # halo radius r             - the radius of the halo, in /4x.
            
            # encoded in the array format in 'locations':
            # [int cx, int cy, float ox, float oy, float majx, float majy, float minx, float miny, float r]
            
            positions = [float(i) for i in line.split(";")]
            head = [positions[0], positions[1]]
            tail = [positions[2], positions[3]]
            minorAxisLength = positions[4]
            haloRadius = positions[5]

            centerRaw = [(head[0] + tail[0]) / 2, (head[1] + tail[1]) / 2]
            centerInt = [centerRaw[0] // 4, centerRaw[1] // 4]
            centerOff = [centerRaw[0] - centerInt[0] * 4, centerRaw[1] - centerInt[1] * 4]
            majorAxis = [(tail[0] - head[0]) / 8, (tail[1] - head[1]) / 8]
            majorMod = sqrt(majorAxis[0] ** 2 + majorAxis[1] ** 2)
            minorUniform = [majorAxis[1] / majorMod, -majorAxis[0] / majorMod]
            minorMod = minorAxisLength / 8
            minorAxis = [minorUniform[0] * minorMod, minorUniform[1] * minorMod]

            locations.append([centerInt[0], centerInt[1],
                              centerOff[0], centerOff[1],
                              majorAxis[0], majorAxis[1],
                              minorMod,
                              haloRadius / 4])
            
    return locations

def generateArchieve(settings, imageFileNames, zipArchieve):
    countDict = {}
    metadata = { 'names' : [] }
    currentImage = 0
    warns = []
    progress = tqdm(imageFileNames)
    for imageFile in progress:
        progress.set_description("{}".format(imageFile))
        
        fullPath = settings["inputImage"] + imageFile
        numpyImage = grayscale(fullPath)
        imageName = os.path.splitext(imageFile)[0]

        # resize image to specified margins.
        width = numpyImage.shape[1]
        height = numpyImage.shape[0]

        padWidth = width + settings["margin"][0] + settings["margin"][2]
        padHeight = height + settings["margin"][1] + settings["margin"][3]

        # repeat generation time
        REPEATGEN = 16
        generalId = 1
        for repeatg in range(REPEATGEN):

            img = torch.from_numpy(numpyImage).reshape(1, 1, height, width)
            zeroTensor = F.pad(img, (settings["margin"][0], settings["margin"][2], settings["margin"][1], settings["margin"][3]), 'reflect')
        
            locations = decode(
                settings["annotation"], imageFile, width, height, settings["iouThreshold"])

            if locations == None:
                continue

            # the reflect padding of the image and heatmap needs the replication of bounding 
            # boxes. the following operation reflects a box for eight times.

            boundboxRepl = []
            for bbox in locations:
                boundboxRepl.append([bbox[0], - bbox[1],                                     bbox[2], -bbox[3], bbox[4], -bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([bbox[0], height // 2 - bbox[1] - 2,                     bbox[2], -bbox[3], bbox[4], -bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([- bbox[0], bbox[1],                                     -bbox[2], bbox[3], -bbox[4], bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([width // 2 - bbox[0] - 2, bbox[1],                      -bbox[2], bbox[3], -bbox[4], bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([width // 2 - bbox[0] - 2, - bbox[1],                    -bbox[2], -bbox[3], -bbox[4], -bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([- bbox[0], - bbox[1],                                   -bbox[2], -bbox[3], -bbox[4], -bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([width // 2 - bbox[0] - 2, height // 2 - bbox[1] - 2,    -bbox[2], -bbox[3], -bbox[4], -bbox[5], bbox[6], bbox[7]])
                boundboxRepl.append([- bbox[0], height // 2 - bbox[1] - 2,                   -bbox[2], -bbox[3], -bbox[4], -bbox[5], bbox[6], bbox[7]])
            locations += boundboxRepl

            for x in range(len(locations)):
                locations[x][0] += settings["margin"][0] // 4
                locations[x][1] += settings["margin"][1] // 4

            # clip image to target size
            if ((padWidth % settings["destinationSize"] != 0) or 
                (padHeight % settings["destinationSize"] != 0)):
                Logger.err("padding cannot fit the destination size")

            # a predefined argumentation (random rotation) in the preparation stage.
            # rotate zeroTensor
            angle = numpy.random.uniform() * 30 - 15
            zeroTensor = rotate(zeroTensor, angle, PaddingMode.MirrorPadding, ResampleMode.Bilinear)

            if len(locations) > 0:
                locs = torch.tensor(locations)
                locs = rotateCoordinates(locs, width // 8, height // 8, angle)
                locations = []
                for it in locs:
                    locations += [[it[0].item(), it[1].item(), it[2].item(), it[3].item(),
                                   it[4].item(), it[5].item(), it[6].item(), it[7].item()]]

            # reshape back.
            zeroTensor = zeroTensor.reshape(padHeight, padWidth)
        
            for x in range(int(padWidth / settings["destinationSize"])):
                for y in range(int(padHeight / settings["destinationSize"])):
                    progress.set_description("{} - Id: {}".format(imageFile, generalId))
                    imageClip = zeroTensor[y * settings["destinationSize"] : (y+1) * settings["destinationSize"],
                        x * settings["destinationSize"] : (x+1) * settings["destinationSize"]]
                
                    key = "{}.{}".format(imageName, generalId)
                    bs = []
                
                    for bbox in locations:
                    
                        if (( (bbox[0] * 4 + bbox[2]) >= x * settings["destinationSize"] and (bbox[0] * 4 + bbox[2]) < (x+1) * settings["destinationSize"]) and \
                            ((bbox[1] * 4 + bbox[3]) >= y * settings["destinationSize"] and (bbox[1] * 4 + bbox[3]) < (y+1) * settings["destinationSize"])) :
                        
                            bsx = [ bbox[0] - x * settings["destinationSize"] // 4, \
                                    bbox[1] - y * settings["destinationSize"] // 4, \
                                    bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]]
                            bs.append(bsx)
                
                    countDict[key] = len(bs)
                    bs = numpy.array(bs)

                    numpy.save("/hy-tmp/temp/confocalCenter/locs/{}.{}.{}.npy".format(imageName, repeatg, generalId), bs)
                    # zipArchieve.write("./{}.{}.locs.npy".format(imageName, generalId), "locs/{}.{}.npy".format(imageName, generalId))
                    # os.remove("./{}.{}.locs.npy".format(imageName, generalId))

                    numpy.save("/hy-tmp/temp/confocalCenter/samples/{}.{}.{}.npy".format(imageName, repeatg, generalId), imageClip.detach().numpy())

                    # compress to the target zip file
                    # zipArchieve.write("./{}.{}.npy".format(imageName, generalId), "samples/{}.{}.npy".format(imageName, generalId))
                    # os.remove("./{}.{}.npy".format(imageName, generalId))

                    metadata["names"].append("{}.{}.{}.npy".format(imageName, repeatg, generalId))
                    generalId += 1
        
        currentImage += 1

    countfile = open("/hy-tmp/temp/confocalCenter/object-count.json", "w+")
    countfile.write(json.dumps(countDict))
    countfile.close()

    metadataFile = open("/hy-tmp/temp/confocalCenter/dataset.json", "w+")
    metadataFile.write(json.dumps(metadata))
    metadataFile.close()

    # zipArchieve.write("./object-count.json", "object-count.json")
    # zipArchieve.write("./dataset.json", "dataset.json")
    # os.remove("./object-count.json")
    # os.remove("./dataset.json")

    for warning in warns:
        Logger.warn(":: preprocess.py :: " + warning)
    

def rotateCoordinates(locs, targetSizeXH, targetSizeYH, angle):
        
    # calculate the center x and ys for central rotation. clockwise.
    locs[:, 0] += ( 0.5 - targetSizeXH )
    locs[:, 1] += ( 0.5 - targetSizeYH )

    sinA = math.sin(- angle * math.pi / 180.0) # clockwise
    cosA = math.cos(- angle * math.pi / 180.0)
    distance = torch.sqrt( torch.pow(locs[:,0],2) + torch.pow(locs[:,1],2) )

    sin = locs[:,1] / distance
    cos = locs[:,0] / distance
    rotsin = sin * cosA + cos * sinA
    rotcos = cos * cosA - sin * sinA
    locs[:,1] = distance * rotsin
    locs[:,0] = distance * rotcos
    
    locs[:, 0] -= ( 0.5 - targetSizeXH )
    locs[:, 1] -= ( 0.5 - targetSizeYH )

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