
import argparse
import importlib
from pprint import pprint
import sys
import os

import torch
import torch.cuda
import torch.jit

from logger import Logger

def parseArguments():
    parser = argparse.ArgumentParser( description = """
        trace.py - generate libtorch capable version of pytorch model.
    """)
    
    parser.add_argument ( "output", type = str,
                          help = "the output .pt file of traced pytorch binary model (for C++ binding)")
    parser.add_argument ( "-a", dest = "modelArchitecture", type = str,
                          help = "the architecture name of the model")
    parser.add_argument ( "-m", type = str, dest = "model",
                          help = "the path to the model file. in .pth format")
    parser.add_argument ( "-s", type = str, dest = "inputShape",
                          help = "the input format of tensor shape, in space-separated strings. e.g. '1 1 64 64'")
    parser.add_argument ( "-gpu", dest = "useGPU", const = True, default = False,
                          help = "whether the trainer detect and use GPUs", action = "store_const")
    parser.add_argument ( "-wrapped", dest = "isWrapped", const = True, default = False,
                          help = "whether the model is trained with wrapped data parallel", action = "store_const")

    args = parser.parse_args()
    return args

@torch.no_grad()
def begin(args):

    modelPy = 'trainer.model.' + args['architecture']
    Logger.info("Loaded Model From: {}".format(modelPy))
    modelLoader = importlib.import_module(modelPy)
    
    model   = modelLoader.model(**modelLoader.modelParams)

    if args['wrapped']:
        model = torch.nn.parallel.DataParallel(model)
    pretrainedModel = args['model']
            
    if os.path.exists(pretrainedModel) == False:
        Logger.err(":: trace.py :: Pretrained Model Does not Exist: {}".format(pretrainedModel))
        sys.exit()
    elif args['useGPU'] and torch.cuda.is_available():
        loadPretrainedCPU(model, pretrainedModel)
        pass
    else:
        if args['useGPU']: Logger.warn(":: trace.py :: The working machine has no GPU available, thus cannot trace GPU models. We switch to CPU mode")
        loadPretrainedCPU(model, pretrainedModel)
    
    wrapper = importlib.import_module('trainer.wrappers.' + args['architecture']).Wrapper(model)

    dummyInp = torch.rand(*args['shape'])
    tracedModel = torch.jit.trace(wrapper, dummyInp)
    dummyOut = wrapper(dummyInp)

    Logger.log("The loaded models accepts Input in {} and Output in {}".format(dummyInp.shape, dummyOut.shape))
    tracedModel.save(args['output'])
    Logger.log("Output saved to {}".format(args['output']))

def loadPretrainedCPU(model, pretrained):
    Logger.warn(":: trace.py :: Loading from Pretrained to CPU Model: {}".format(pretrained))
    with open(pretrained, "rb") as f:

        params = torch.load(f, map_location = 'cpu')
        model.load_state_dict(params)

def loadPretrained(model, pretrained):
    Logger.warn(":: trace.py :: Loading from Pretrained: {}".format(pretrained))
    with open(pretrained, "rb") as f:

        params = torch.load(f)
        model.load_state_dict(params)
    
def main(args):
    Logger.info(":: trace.py :: convert .pth models to .pt models :::::::::::::::::::::::::::::")
    Logger.info(":: trace.py :: arguments :::::::::::::::::::::::::::::::::::::::::::::::::::::")
    
    settings = {
        'useGPU'            : args.useGPU,
        'wrapped'           : args.isWrapped,
        'model'             : args.model,
        'architecture'      : args.modelArchitecture,
        'output'            : args.output,
        'shape'             : [int(i) for i in args.inputShape.split(" ")]
    }
    
    pprint(settings, indent = 4)

    Logger.info(":: trace.py :: generating models :::::::::::::::::::::::::::::::::::::::::::::")

    begin(settings)

    Logger.info(":: train.py :: model generation completed ::::::::::::::::::::::::::::::::::::")

if __name__ == "__main__":
    args = parseArguments()
    main(args)
