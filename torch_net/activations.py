"""
Author: Liam Laidlaw
Purpose: Additonal Activation Functions for cvnns.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F


def modrelu(x, bias: float=1, epsilon: float=1e-8):
    magnitude = x.abs()
    activated_magnitude = F.relu(magnitude + bias)
    nonzero_magnitude = magnitude + epsilon
    return activated_magnitude * (magnitude / nonzero_magnitude)


def zrelu(x):
    if not x.is_complex():
        raise TypeError(f"Input must be a complex tensor. Got type {input.dtype}")

    angles = torch.angle(x)
    mask = (angles >= 0) & (angles <= torch.pi/2)
    zeros = torch.zeros_like(x)
    return torch.where(mask, x, zeros)



def complex_cardioid(x):
    angle = torch.angle(x)
    return 0.5 * (1 + torch.cos(angle)) * x


class ModReLU(nn.Module):
    def __init__(self, bias: float=1, dtype=torch.float32):
        super(ModReLU, self).__inti__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=dtype)) # make bias learnable


    def forward(self, input):
        if not input.is_complex():
            raise TypeError(f"Input must be a complex tensor. Got type {input.dtype}")
        return modrelu(input, self.bias)


class ZReLU(nn.Module):






