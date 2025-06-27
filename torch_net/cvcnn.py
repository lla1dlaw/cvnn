"""
Author: Liam Laidlaw
Purpose: Complex Valued Convolutional Neural Network

Utilizes Sébastien M. Popoff's complexPyTorch framework. Available: https://github.com/wavefrontshaping/complexPyTorch/tree/master
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from typing import Any 


class MNIST_CVCNN(nn.Module):
    """Designed for MNIST

    """
    def __init__(self, hidden_activation: str | Any, output_activation: Any | None):
        super(MNIST_CVCNN, self).__init__()

        function_dispatcher = {
            'crelu': complex_relu,
            'cart_relu': complex_relu,
            # add the other functions once you write them
        }

        if output_activation != None:
            self.output_activation = function_dispatcher.get(output_activation, output_activation) # if output_activation is specified, try and get it from the dispatcher.

        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.bn1  = ComplexBatchNorm2d(10)
        self.hidden_activation = function_dispatcher.get(hidden_activation, hidden_activation)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.bn2 = ComplexBatchNorm2d(20)
        self.fc1 = ComplexLinear(4*4*20, 500)
        self.fc2 = ComplexLinear(500, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hidden_activation(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hidden_activation(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,4*4*20)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x



