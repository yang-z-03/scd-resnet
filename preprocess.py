
# pytorch references ...
# and pytorch multi-gpu support

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
from math import sqrt, exp
from evaluations.intersection import cornerThresholdRadius
from enum import Enum
from logger import Logger
from tqdm import tqdm
import json
import sys
import importlib

def parseArguments():
    parser = argparse.ArgumentParser( description = """
        preprocess.py - sample preprocess executable for neural network training:
        raw full slide images will be clipped and transformed into a specified size
        and labelimg format of annotations will be decoded to corresponding heatmap
        in the form of numpy array savings.
    """)
    
    parser.add_argument ( "outputZipPath", type = str,
                          help = "the location to place the output zipped samples.")
    parser.add_argument ( "-i", dest = "inputImage",
                          help = "input image folder.", type = str )
    parser.add_argument ( "-a", dest = "annotation", type = str, 
                          help = "input annotation folder in labelImg YOLO format.")
    parser.add_argument ( "-s", dest = "destinationSize",
                          help = "destination image size.",
                          default = 512, type = int )
    parser.add_argument ( "-t", dest = "iouThreshold",
                          help = "IoU threshold for gaussian radius determination.",
                          default = 0.7, type = float )
    parser.add_argument ( "-v", dest = "verbal",
                          help = "display the heatmap and clip result (debug).",
                          const = True, default = False, action = 'store_const' )
    parser.add_argument ( "-m", dest = "margin", default = "0 0 0 0", 
                          type = str, 
                          help = """the border margin to fill blank, in the form of 
                          'leftMargin topMargin rightMargin bottomMargin'.""")

    # the profile module for a preprocessor, must implement the function:
    # def generateArchieve(settings, imageFileNames, zipArchieve).
    parser.add_argument ( "-p", dest = "profile", type = str, 
                          help = """the preprocess profile module""")

    args = parser.parse_args()
    return args

def main(args):
    Logger.info(":: preprocess.py :: preprocess and generate samples from whole slide images ::")
    Logger.info(":: preprocess.py :: arguments ::::::::::::::::::::::::::::::::::::::::::::::::")
    
    settings = {
        'outputPath'        : args.outputZipPath,
        'inputImage'        : args.inputImage,
        'annotation'        : args.annotation,
        'destinationSize'   : args.destinationSize,
        'margin'            : [int(i) for i in args.margin.split(" ")],
        'iouThreshold'      : args.iouThreshold,
        'verbal'            : bool(args.verbal),
        'profile'           : args.profile
    }
    pprint.pprint(settings, indent = 4)

    Logger.info(":: preprocess.py :: generating image clips :::::::::::::::::::::::::::::::::::")

    imageFileNames = os.listdir(settings["inputImage"])
    imageFileNames = sorted(imageFileNames, key = lambda i:int(re.match(r'(\d+)',i).group()))
    
    totalImage = len(imageFileNames)

    zip = zipfile.ZipFile(settings["outputPath"], "w", zipfile.ZIP_DEFLATED)

    profileModule = importlib.import_module(settings["profile"])
    profileModule.generateArchieve(settings, imageFileNames, zip)

    Logger.info(":: preprocess.py :: task completed successfully ::::::::::::::::::::::::::::::")

    zip.close()
    return args

if __name__ == "__main__":
    args = parseArguments()
    main(args)
