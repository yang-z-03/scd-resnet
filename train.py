
import argparse
import json
import os
from pprint import pprint
import random
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy
import torch
import torch.cuda
import torch.distributed as dist
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed as utilsDataDist
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import sgd
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from logger import Logger
from configuration import defaultConfig
from models.networkFactory import NetworkFactory

GPUCOUNT = 0

def parseArguments():
    parser = argparse.ArgumentParser( description = """
        train.py - train neural networks with a given set of configuration.
    """)
    
    parser.add_argument ( "configuration", type = str,
                          help = """the path to the configuration file. for detailed file 
                          format, see explanations in ./readme.md """)
    parser.add_argument ( "-gpu", dest = "useGPU", const = True, default = False,
                          help = "whether the trainer detect and use GPUs", action = "store_const")
    parser.add_argument ( "-debug", dest = "debug", const = True, default = False,
                          help = "enable debug features, including visualization etc.", action = "store_const")

    # the --local-rank argument is passed by the torch.distributed.launch program.
    # for identification of multi-gpu training process index.
    parser.add_argument ( "--local_rank", default = -1, type = int, dest = "localRank",
                          help = "local process index for torch.distributed.launch")

    args = parser.parse_args()
    return args

def begin(args):

    # get information of available gpus and allocate them.
        
    GPUCOUNT = torch.cuda.device_count()
    localRank = -1
    if args["useGPU"]:
        if GPUCOUNT > 0 and torch.cuda.is_available():
            pass
        else:
            GPUCOUNT = 0
            Logger.warn(":: train.py :: No GPU Available on Current Machine, Run on CPU Instead.")
        
        args["useGPU"] = (GPUCOUNT != 0)
    
        if dist.is_nccl_available() is not True:
            Logger.err(":: train.py :: The NCCL Backend are Not Set Up on This Machine")
            sys.exit()

        # initialize the multiprocess group through nccl backends.
        dist.init_process_group("nccl")
        localRank = args["localRank"]
    
    with open(args["config"], "r") as f:
        configs = json.load(f)
        defaultConfig.updateConfig(configs)
    pprint(defaultConfig.config, indent = 4)
    Logger.info(":: train.py :: configuration :::::::::::::::::::::::::::::::::::::::::::::::::")

    if args["useGPU"]:    
        torch.cuda.set_device(localRank)

    trainFactory = NetworkFactory(args["useGPU"])

    trainFactory.beginTraining(localRank)
    
def main(args):
    Logger.info(":: train.py :: trainer program of neural networks ::::::::::::::::::::::::::::")
    Logger.info(":: train.py :: arguments :::::::::::::::::::::::::::::::::::::::::::::::::::::")
    
    settings = {
        'config'            : args.configuration,
        'useGPU'            : args.useGPU,
        'localRank'         : args.localRank,
        'debug'             : args.debug
    }

    defaultConfig.update('useGPU', args.useGPU)
    pprint(settings, indent = 4)

    Logger.info(":: train.py :: trainer task begin ::::::::::::::::::::::::::::::::::::::::::::")

    begin(settings)

    Logger.info(":: train.py :: trainer task completed ::::::::::::::::::::::::::::::::::::::::")

if __name__ == "__main__":
    args = parseArguments()
    main(args)