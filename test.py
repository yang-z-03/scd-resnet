
import numpy as numpy
from scipy.optimize import curve_fit as curveFit, fmin

import json
from math import ceil, sqrt
from PIL import Image
from logger import Logger
import torch
import torch.nn.functional as F
import torch.jit
from datasets.argumentations import normalize

def gauss2(x, a1, m1, s1, a2, m2, s2): return a1 * numpy.exp(-((x-m1)/s1)**2) + a2 * numpy.exp(-((x-m2)/s2)**2)

INPUTSIZE = 512
PADDINGSIZE = 64
DOWNSAMPLERATIO = 4
BATCHSIZE = 24

def grayscale(path) -> numpy.ndarray:
    image = Image.open(path)

    # convert into grayscale image.
    colorImage = numpy.array(image)
    r = colorImage[:,:,0]
    g = colorImage[:,:,1]
    b = colorImage[:,:,2]

    # in opencv, the grayscale image will be rounded
    numpyImage = numpy.round( 0.1140 * r + 0.5870 * g + 0.2989 * b )

    return numpyImage, colorImage

def loadPretrainedCPU(model, pretrained):
    Logger.warn("Loading from Pretrained: {}".format(pretrained))
    with open(pretrained, "rb") as f:
        params = torch.load(f, map_location = 'cpu')
        model.load_state_dict(params)

def analyseImages(model, fullPath) -> list :
    numpyImage, colorImg = grayscale(fullPath)

    # resize image to specified margins.
    width = numpyImage.shape[1]
    height = numpyImage.shape[0]

    clipHorizontal = ceil((width - 2 * PADDINGSIZE) / (INPUTSIZE - 2 * PADDINGSIZE))
    clipVertical = ceil((height - 2 * PADDINGSIZE) / (INPUTSIZE - 2 * PADDINGSIZE))

    resizeW = (INPUTSIZE - 2 * PADDINGSIZE) * clipHorizontal + 2 * PADDINGSIZE
    resizeH = (INPUTSIZE - 2 * PADDINGSIZE) * clipVertical + 2 * PADDINGSIZE

    if (resizeW - width) % 2 != 0: resizeW += 1
    if (resizeH - height) % 2 != 0: resizeH += 1
    padLR = (resizeW - width) // 2
    padTB = (resizeH - height) // 2

    img = torch.from_numpy(numpyImage).reshape(1, 1, height, width)
    zeroTensor = F.pad(img, (padLR, padLR, padTB, padTB), 'reflect')

    # here, we will manually fix an issue to ensure complete similarity between opencv and
    # pytorch. in opencv, the mirror padding 'reflect' act like this.
    #
    #    |-----------------------|
    #    |           |           |
    #    |        |--|--|        |
    #   [0] ... [63] | [64] ... [127] (opencv)
    #
    # while in torch,
    #
    #    |------------------------------------|
    #    |        |------------------|        |
    #    |        |    |--------|    |        |
    #   [0] ... [62] [63] [64] [65] [66] ... [128] (torch)
    # 
    # here, we will choose the opencv mode.
    
    for x in range(0, 64):
        zeroTensor[0,0,:,x] = zeroTensor[0,0,:,127 - x]
    for x in range(3136, 3200):
        zeroTensor[0,0,:,x] = zeroTensor[0,0,:,6271 - x]
        
    zeroTensor = zeroTensor.reshape(1, resizeH, resizeW)
        
    imageClip = []
    for x in range(clipHorizontal):
        for y in range(clipVertical):
            imageClip += [normalize(zeroTensor[:, y * (INPUTSIZE - 2 * PADDINGSIZE) : y * (INPUTSIZE - 2 * PADDINGSIZE) + INPUTSIZE,
                                               x * (INPUTSIZE - 2 * PADDINGSIZE): x * (INPUTSIZE - 2 * PADDINGSIZE) + INPUTSIZE]).float()]
        
    results = []

    # process the image clips in specified batch size.
    for i in range(ceil(len(imageClip) / BATCHSIZE)):
        if i == len(imageClip) - 1:
            input = torch.stack(imageClip[i * BATCHSIZE : len(imageClip)], 0)
        else:
            input = torch.stack(imageClip[i * BATCHSIZE : (i+1) * BATCHSIZE], 0)
            
        # validation mode, returns a decoded result.
        # ctScores, _, ctY, ctX, sizes, _ = model(input)
        ctScores, _, ctY, ctX, majX, majY, minL, rad, offX, offY = model(input)
            
        for item in range(len(ctScores)):
            threshold = ctScores[item] > 0.3
            results.append([
                ctX[item][threshold], ctY[item][threshold],
                offX[item][threshold], offY[item][threshold],
                majX[item][threshold], majY[item][threshold],
                minL[item][threshold], rad[item][threshold]
            ])

    id = 0
    detections = []

    for x in range(clipHorizontal):
        for y in range(clipVertical):
            decode = results[id]
            centerX, centerY, offsetX, offsetY, majorX, majorY, minorL, radius = decode

            detectionNum = len(centerX)
            for brid in range(detectionNum):
                dminl = minorL[brid].item() * 4
                halo = radius[brid].item() * 4

                ratio = (halo - dminl) / (2 * dminl)
                
                cx = centerX[brid]
                cy = centerY[brid]

                # if cx < PADDINGSIZE // (2 * DOWNSAMPLERATIO) or cx >= PADDINGSIZE - PADDINGSIZE // (2 * DOWNSAMPLERATIO) or \
                #    cy < PADDINGSIZE // (2 * DOWNSAMPLERATIO) or cy >= PADDINGSIZE - PADDINGSIZE // (2 * DOWNSAMPLERATIO):
                #     continue

                tupleResult = [int(x * (INPUTSIZE - 2 * PADDINGSIZE) - padLR + centerX[brid].item() * 4 + offsetX[brid].item()), 
                               int(y * (INPUTSIZE - 2 * PADDINGSIZE) - padTB + centerY[brid].item() * 4 + offsetY[brid].item()), ratio]
                detections += [tupleResult]

            id += 1
    
    return detections

########################### PLACE YOUR .PT FILE HERE ###########################
model = torch.jit.load('xxx.pt')
model.eval()

def distance(x1, y1, x2, y2):
    dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return sqrt(dist)


'''
We assume imgs contains a list of file paths to your test images, each image is
assumed to be 3092 x 2056 in size.
--------------------------------------------------------------------------------
    register = []
    for img in imgs:
        sperms = analyseImages(model, img)
        for i in range(len(sperms)): sperms[i] += [img]
        register += sperms

    validSample = []
    for ctx, cty, rhr, uid in register:
        flag = True
            
        if ctx < 0 or cty < 0 or ctx >= 3072 or cty >= 2056:
            continue 

        if flag: validSample += [[ctx, cty, rhr, uid]]
--------------------------------------------------------------------------------
By now, validSample contains all the x, y, Rhr and the file name in the image.
--------------------------------------------------------------------------------
    xs = [(x - 25)/ 100 for x in range(150)]
    ys = ...

    pbounds2 = ([0, -0.25, 0, 0, 0, 0], [1, 0.33, 0.2, 1, 1.25, 1])
    popt2, pcov2 = curveFit(gauss2, xs, ys, bounds = pbounds2, maxfev = 5000)
--------------------------------------------------------------------------------
ys contains the histogram data to the relative halo radius. it is a length-150
vector, containing frequencies for Rhr = -0.25 to 1.25. (interval 0.01). Fit the
Gaussian distribution, and you will get popt2 as [a1, m1, s1, a2, m2, s2].
'''
