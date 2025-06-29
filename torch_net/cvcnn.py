"""
Author: Liam Laidlaw
Purpose: Complex Valued Convolutional Neural Network

Utilizes Sébastien M. Popoff's complexPyTorch framework. Available: https://github.com/wavefrontshaping/complexPyTorch/tree/master
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU, ComplexMaxPool2d
from activations import ModReLU, ZReLU, ComplexCardioid
from typing import Any 


class MNIST_CVCNN(nn.Module):
    def __init__(self, hidden_activation: str | Any, output_activation: str | None = None):
        """Designed for MNIST
        Args:
            hidden_activation (str | Any): The activation function to use for the hidden layer/s. Can be either a string (the name of the function) or the module itself. 
            output_activation (str | None, optional): The activation function to apply to the output. Same useage as hidden_activaiton. Defaults to None. 
        Returns:
            None
        """
        super(MNIST_CVCNN, self).__init__()

        function_dispatcher = {
            'modrelu': ModReLU,
            'zrelu': ZReLU,
            'crelu': ComplexReLU,
            'complex_cardioid': ComplexCardioid,
        }

        if output_activation != None:
            self.output_activation = function_dispatcher.get(output_activation, output_activation) # if output_activation is specified, try and get it from the dispatcher.
        layers = [
            ComplexConv2d(1, 10, 5, 1),
            ComplexBatchNorm2d(10),
            function_dispatcher.get(hidden_activation, hidden_activation)(), # if the hidden activation string is not found, the argument is assumed to be the module itself
            ComplexConv2d(10, 20, 5, 1),
            ComplexBatchNorm2d(20),
            function_dispatcher.get(hidden_activation, hidden_activation)(), 
            ComplexLinear(4*4*20, 500),
            ComplexLinear(500, 10)
        ]
        self.layers = nn.Sequential(*layers)


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



