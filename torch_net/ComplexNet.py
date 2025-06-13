import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Linear as ComplexLinear

# from complextorch.nn.modules.linear import Linear as ComplexLinear
# from complextorch.nn import CVLinear as ComplexLinear
from complexPyTorch.complexLayers import ComplexLinear 
from complextorch.nn.modules.conv import Conv2d as ComplexConv2d
from complextorch.nn.modules.batchnorm import BatchNorm2d as ComplexBatchNorm2d
from complextorch.nn.modules.activation import zReLU
from complextorch.nn.modules.softmax import CVSoftMax
from complextorch.nn.modules.batchnorm import BatchNorm1d as ComplexBatchNorm1d

class ComplexNet(nn.Module):
    def __init__(self, linear_shape: list[int], in_size: int, out_size: int, conv_shape: list[int] = None):
        """Complex Neural Network Class

        Args:
            linear_shape (list[int]): List of integers representing the shape of hidden linear layers.
            in_size (int): Dimension of the input tensor.
            out_size (int): Dimension of the output tensor.
            conv_shape (list[int], optional): List of integers representing the shape of convolutional linear layers. Defaults to None.
        """
        super(ComplexNet, self).__init__()

        if conv_shape: # If conv_shape is provided, use it
            pass # add conv layers here

        prev_outsize = in_size
        self.layers = []
#        self.layers.append(nn.Flatten(1))
        for shape in linear_shape:
            self.layers.append(ComplexLinear(prev_outsize, shape)) # hidden layers
            prev_outsize = shape
            self.layers.append(zReLU())
            self.layers.append(ComplexBatchNorm1d(shape)) # batch norm after each hidden layer
        self.layers.append(ComplexLinear(prev_outsize, out_size)) # output layer
        self.layers.append(CVSoftMax(1))
        self.layers = nn.ModuleList(self.layers)


    def forward(self,x):
        print(f"\n\nInput shape: {x.shape}")
        print(f"Sample input: {x[0][0][0]}")
        print(f"Sample input type: {x[0][0][0].dtype}")
        x = torch.flatten(x, start_dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Layer{i} output shape: {x.shape} ")
            print(f"Sample layer{i} datapoint: {x[0][0]}")
        return x
