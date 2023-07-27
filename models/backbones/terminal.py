
class BackboneTerminal(object):

    # interface of hourglass terminal. optional implement the initializer function
    # >> func (model:torch.nn.Module)
    # the makeLayer function
    # >> func (predictionDimension:int, currentDimension:int, outputDimension:int)
    # and the process function
    # >> func (input:Tensor, module:Module, *xs, **kwargs)

    def __init__(self, name, initializerFunction = None, makeLayerFunction = None, process = None):
        self.name = name
        self.makeLayer = makeLayerFunction
        self.initializer = initializerFunction
        self.process = process