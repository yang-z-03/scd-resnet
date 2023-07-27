
import torch
from torch import nn
from torch.autograd import Function

import topPool, bottomPool, leftPool, rightPool

class TopPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = topPool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = topPool.backward(input, grad_output)[0]
        return output

class BottomPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = bottomPool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = bottomPool.backward(input, grad_output)[0]
        return output

class LeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = leftPool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = leftPool.backward(input, grad_output)[0]
        return output

class RightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = rightPool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = rightPool.backward(input, grad_output)[0]
        return output

class TopPool(nn.Module):
    def forward(self, x):
        return TopPoolFunction.apply(x)

class BottomPool(nn.Module):
    def forward(self, x):
        return BottomPoolFunction.apply(x)

class LeftPool(nn.Module):
    def forward(self, x):
        return LeftPoolFunction.apply(x)

class RightPool(nn.Module):
    def forward(self, x):
        return RightPoolFunction.apply(x)