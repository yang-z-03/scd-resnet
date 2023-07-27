
import os

# the configuration project-wide. declaring an editable configuration profile that can
# be changed by user-defined config.json.

class Configuration:
    def __init__(self):
        self.config = {}

        self.config["datasetName"]              = None    # e.g. 'confocalCenter'
        self.config["modelName"]                = None    # e.g. 'corner'
        self.config["trainName"]                = None    # e.g. 'v1'

        # Training Config
        
        self.config["learningRate"]             = 0.00025
        self.config["learningRateDecay"]        = [80000]
        self.config["learningRateDecayRate"]    = [10]

        self.config["currentIter"]              = 0
        self.config["iterations"]               = 117000
        self.config["validation"]               = 200
        self.config["snapshot"]                 = 2000
        self.config["batchSize"]                = 32
        self.config["validationBatchSize"]      = 160
        
        self.config["naming"]                   = "{modelName}.{trainName}.{currentIter}.pth"
        self.config["namingOptimizer"]          = "{naming}.{optimizer}.pth"
        self.config["pretrain"]                 = None    # e.g. 'cornernet.v1.35000'
        self.config["optimizer"]                = "adam"

        # Directories

        self.config["dirData"]                  = "trainer.dataset.{datasetName}"
        self.config["dirModel"]                 = "trainer.model.{modelName}"
        self.config["dirTemp"]                  = "/temp/"
        self.config["dirPretrain"]              = "/pretrain/"
        self.config["dirConfig"]                = "/configs/"
        self.config["dirResult"]                = "/results/"
        self.config["dirDataset"]               = "/datasets/"
        self.config["dirDatafile"]              = "{dirDataset}{datasetName}.d"
        self.config["dirDataSplitProfile"]      = "{dirDataset}{datasetName}.split.json"
        self.config["useGPU"]                   = False

    @property
    def pretrain(self):
        if self.config["pretrain"] is not None:
            return self.config["dirPretrain"] + self.config["pretrain"]
        else: 
            return None

    @property
    def datasetName(self):
        return self.config["datasetName"]

    @property
    def modelName(self):
        return self.config["modelName"]
    
    @property
    def trainName(self):
        return self.config["trainName"]
    
    @property
    def learningRate(self):
        return self.config["learningRate"]
    
    @property
    def learningRateDecay(self):
        return self.config["learningRateDecay"]

    @property
    def learningRateDecayRate(self):
        return self.config["learningRateDecayRate"]
    
    @property
    def totalIterations(self):
        return self.config["iterations"]
    
    @property
    def snapshotFrequency(self):
        return self.config["snapshot"]
    
    @property
    def validationFrequency(self):
        return self.config["validation"]

    @property
    def batchSize(self):
        return self.config["batchSize"]
    
    @property
    def validationBatchSize(self):
        return self.config["validationBatchSize"]
    
    @property
    def currentIteration(self):
        return self.config["currentIter"]
    
    @property
    def naming(self):
        return self.config["naming"].format(**self.config)

    @property
    def optimizer(self):
        return self.config["optimizer"].format(**self.config)

    @property
    def namingOptimizer(self):
        return self.config["namingOptimizer"]
    
    @property
    def dirData(self):
        return self.config["dirData"].format(**self.config)
    
    @property
    def dirModel(self):
        return self.config["dirModel"].format(**self.config)

    @property
    def dirTemp(self):
        if not os.path.exists(self.config["dirTemp"]):
            os.makedirs(self.config["dirTemp"])
        return self.config["dirTemp"]

    @property
    def dirResult(self):
        if not os.path.exists(self.config["dirResult"]):
            os.makedirs(self.config["dirResult"])
        return self.config["dirResult"]

    @property
    def dirConfig(self):
        if not os.path.exists(self.config["dirConfig"]):
            os.makedirs(self.config["dirConfig"])
        return self.config["dirConfig"]

    @property
    def dirDatafile(self):
        return self.config["dirDatafile"].format(**self.config)

    @property
    def dirDataSplitProfile(self):
        return self.config["dirDataSplitProfile"].format(**self.config)
    
    def useGPU(self):
        return self.config["useGPU"]

    def updateConfig(self, configObj):
        for key in configObj:
            if key in self.config:
                self.config[key] = configObj[key]
    
    def updateIteration(self, iter):
        self.config["currentIter"] = iter
    
    def update(self, configName, value):
        self.config[configName] = value

defaultConfig = Configuration()