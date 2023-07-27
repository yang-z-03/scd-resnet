
import torch
import torch.nn

class Wrapper(torch.nn.Module):

    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
    
    def forward(self, inp):
        decode = self.model(inp, decode = True)
        decode.pop()
        regression = decode.pop()
        offset = decode.pop()
        decode.append(regression[:,:,0])     # major axis x.
        decode.append(regression[:,:,1])     # major axis y.
        decode.append(regression[:,:,2])     # minor axis length.
        decode.append(regression[:,:,3])     # radius
        decode.append(offset[:,:,0])
        decode.append(offset[:,:,1])
        
        return torch.stack(decode)