
# pytorch references ...
# and pytorch multi-gpu support

from asyncio.log import logger
from random import shuffle
import numpy
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
import torch.distributed as dist
import torch.multiprocessing as multiproc

import os
import json
import importlib
from configuration import defaultConfig
from logger import Logger, monitorStdOutStream
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.backbones.utility import reshapeGatherFeatures

torch.random.manual_seed(42)

class NetworkFactory(object):

    GPUCOUNT = 0
    
    # important note: only initialize ONE network factory in a training task. it will run
    # the model with distributed data parallel if 'useGPU' is set to True, this will by
    # default takes all the available GPUs on one machine.

    def __init__(self, useGPU):
        super(NetworkFactory, self).__init__()

        self.useGPU = useGPU
        self.GPUCOUNT = torch.cuda.device_count()

        modelPy = defaultConfig.dirModel
        Logger.info("Loaded Model From: {}".format(modelPy))
        modelLoader = importlib.import_module(modelPy)

        self.model   = modelLoader.model(**modelLoader.modelParams)
        self.loss    = modelLoader.loss
        self.evaluation = modelLoader.evaluation
        self.evalExpr = modelLoader.expression

        dataPy = defaultConfig.dirData
        Logger.info("Loaded Dataset File From: {}".format(dataPy))
        dataLoader = importlib.import_module(dataPy)
        
        if os.path.exists(defaultConfig.dirDataSplitProfile):
            with open(defaultConfig.dirDataSplitProfile, "r") as file: 
                jsonObj = json.loads(file.read())
                self.dataset = dataLoader.dataset(defaultConfig.dirDatafile, useGPU, jsonObj)
        else:
            self.dataset = dataLoader.dataset(defaultConfig.dirDatafile, useGPU, None)

        totalParameters = 0
        for params in self.model.parameters():
            paramCount = 1
            for x in params.size():
                paramCount *= x
            totalParameters += paramCount
        Logger.log("Parameter Count: {}".format(totalParameters))
        self.parameterCount = totalParameters

        if defaultConfig.optimizer == "adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )

        elif defaultConfig.optimizer == "sgd":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr = defaultConfig.learningRate, 
                momentum = 0.9, weight_decay = 0.0001
            )

        else:
            Logger.err(":: networkFactory.py :: Unknown Optimizer '{}', Currently Support 'sgd' or 'adam'".format(defaultConfig.optimizer))
            sys.exit()

    @property
    def isGPU(self):
        return self.useGPU
    
    def beginTraining(self, localRank):

        # we set the sampler and loader's shuffle to False, for we apply shuffle 
        # already in the dataset.

        if self.useGPU:
            Logger.info(":: networkFactory.py :: Begin Training Task on Local Device {}".format(localRank))
            sampler = utilsDataDist.DistributedSampler(self.dataset, drop_last = True, shuffle = False)
            trainLoader = DataLoader( self.dataset, batch_size = defaultConfig.batchSize,
                                      sampler = sampler, drop_last = True, shuffle = False)
        else:
            trainLoader = DataLoader( self.dataset, batch_size = defaultConfig.batchSize, drop_last = True, shuffle = False )

        Logger.log("Loaded Dataset Loader: {}".format(defaultConfig.datasetName))
        Logger.info("Loaded with Training Samples: {}".format(self.dataset.__len__()))
            
        # update the learning rate if trained from the midtime
        learningRate = defaultConfig.learningRate
        if defaultConfig.currentIteration > 0:
            for t in range(1, defaultConfig.currentIteration):
                if t in defaultConfig.learningRateDecay:
                    id = defaultConfig.learningRateDecay.index(t)
                    learningRate /= defaultConfig.learningRateDecayRate[t]
                
            self.loadParameters()
            self.setLearningRate(learningRate)
            
        if self.useGPU:
            self.cuda()
            if self.GPUCOUNT > 1:
                
                # enable the sync batch normalization to ensure batch size across all 
                # the gpu devices.

                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters = True )
        
        else: self.model = torch.nn.parallel.DataParallel(self.model)
        
        pretrainedModel = defaultConfig.pretrain
        if pretrainedModel is not None:
            if not os.path.exists(pretrainedModel):
                Logger.err(":: networkFactory.py :: Pretrained Model Does not Exist")
                sys.exit()
            print("loading from pretrained model")
            self.loadPretrained(pretrainedModel)

        self.trainMode()

        startIteration = defaultConfig.currentIteration
        it = startIteration

        learningRateDecay = defaultConfig.learningRateDecay
        learningRateDecayRate = defaultConfig.learningRateDecayRate
        lossSave = []
        evalResult = []

        evalResult += ["Experiment: {}".format(defaultConfig.trainName) + '\n']
        evalResult += ["Parameter Count: {}".format(self.parameterCount) + '\n']

        with monitorStdOutStream() as saveStdOut:
            with tqdm( total = defaultConfig.totalIterations - startIteration,
                       file = saveStdOut, ncols = 100 ) as pbar:
                
                finished = False
                while not finished:

                    for batchId, data in enumerate(trainLoader):
                        defaultConfig.updateIteration(it)
                        it += 1

                        # for training samples, they were on CPU before every epoch, we need
                        # to transfer them onto the GPU memory and transfer them back after
                        # training, thus saving GPU space.

                        # train the network.
                        loss, lossStatsT = self.train(**data)
                        pbar.set_description('Loss:' + format(loss.item(), '-10.4f'))
                        lossSave += [it, loss.item()]
                        lossSave += [x.item() for x in lossStatsT]

                        # validate the network.
                        if it % defaultConfig.validationFrequency == 0:

                            # we assume that validation data (only 10%) has been transferred to
                            # GPU during its initialization.
                            
                            trainPerf = []
                            trainResults, trainPreds = self.validate(**data)
                            trainPerf += [trainResults]
                            evalTr = "[Tr] {}:     ".format(format(it, "7d")) + self.evalExpr(trainPerf)

                            with torch.no_grad():
                                validData = self.dataset.getValidationSet()
                                batches = []
                                
                                for item in validData:
                                    
                                    results, preds = self.validate(**item)
                                    batches += [results]
                                    
                                    # preds['size'] = reshapeGatherFeatures(preds['size'], item['ys'][5])
                                    # loss, lossStats = self.loss([preds], item['ys'])
                                    # loss = loss.mean()
                                    # validationLoss += loss.item()

                                    # pred += [preds['heatmap'], preds['tl'], preds['br']]

                            evalr = "[It] {}:     ".format(format(it, "7d")) + self.evalExpr(batches)

                            evalResult += [evalTr + '\n' + evalr + "\n"]
                            Logger.infoGreen(evalTr)
                            Logger.info(evalr)
                        
                        # save parameter cache.
                        if it % defaultConfig.snapshotFrequency == 0:
                            self.saveParameters()
                            numpyLoss = numpy.array(lossSave)
                            dim = 2 + len(lossStatsT)
                            saveData = numpy.zeros((len(numpyLoss[0::dim]), dim))

                            for i in range(dim):
                                saveData[:, i] = numpyLoss[i::dim]
                            numpy.savetxt(defaultConfig.dirResult + "losses.{}.{}.txt".format(defaultConfig.trainName, it), saveData, delimiter = ',', fmt = '%.5f')
                            lossSave = []

                        pbar.update()
                        pass
                        
                        if len(learningRateDecay) >= 1:
                            if it == learningRateDecay[0]:
                                learningRate /= learningRateDecayRate[0]
                                self.setLearningRate(learningRate)
                            
                                learningRateDecayRate.pop(0)
                                learningRateDecay.pop(0)

                        if it >= defaultConfig.totalIterations:
                            finished = True
                            break
        
        with open(defaultConfig.dirResult + "evals.{}.txt".format(defaultConfig.trainName), "w") as evalText:
            evalText.writelines(evalResult)
    
    def cuda(self):
        self.model = self.model.cuda()

    def trainMode(self):
        self.model.train()

    def evalMode(self):
        self.model.eval()

    def _passParams(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs) 
        loss, lossStats = self.loss(preds, ys)
        return loss, lossStats

    def train(self, xs, ys, **kwargs):
        self.optimizer.zero_grad()
        loss, lossStats = self._passParams(xs, ys, decode = False)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss, lossStats

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            decodeResult = self.model(*xs, **kwargs, decode = True)
            
            # now that we obtained the decoded result, we will pass them to the evaluation
            # functions to generate a set of evaluation metrics.
            return self.evaluation(xs, ys, *decodeResult)

    def setLearningRate(self, lr):
        Logger.warn(":: networkFactory.py :: Setting Learning Rate to: {}".format(lr))
        for parameterGroup in self.optimizer.param_groups:
            parameterGroup["lr"] = lr

    def loadPretrained(self, pretrained):
        Logger.warn(":: networkFactory.py :: Loading from Pretrained: {}".format(pretrained))
        with open(pretrained, "rb") as f:
            params = torch.load(f) if self.useGPU else torch.load(f, map_location = 'cpu')
            self.model.load_state_dict(params)

    def loadPretrainedCPU(self, pretrained):
        Logger.warn(":: networkFactory.py :: Loading from Pretrained: {}".format(pretrained))
        with open(pretrained, "rb") as f:
            params = torch.load(f, map_location = 'cpu')
            self.model.load_state_dict(params)

    def loadParameters(self):
        cacheFile = defaultConfig.dirTemp + defaultConfig.naming
        Logger.warn(":: networkFactory.py :: Loading Model from Cached: {}".format(cacheFile))
        with open(cacheFile, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def saveParameters(self):
        cacheFile = defaultConfig.dirTemp + defaultConfig.naming
        Logger.warn(":: networkFactory.py :: Saving Model to {}".format(cacheFile))
        with open(cacheFile, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)