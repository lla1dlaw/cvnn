"""
Author: Liam Laidlaw
Purpose: Classes for defining real and complex valued residual blocks. 
"""

import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexAvgPool2d, ComplexMaxPool2d
from activations import ModReLU, ZReLU, ComplexCardioid, CReLU, AbsSoftmax


class LearnImagResidualBlock(nn.Module):
    """
    A real-valued residual block to learn the imaginary component from a real-valued input.
    This block operates on real tensors to generate the initial imaginary part for the complex network.
    Structure: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LearnImagResidualBlock, self).__init__()
        # First part of the residual connection
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        
        # Second part of the residual connection
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=True)
        
        # Shortcut connection to match dimensions if in_channels != out_channels
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A real-valued input tensor.
        Returns:
            torch.Tensor: A real-valued tensor representing the learned imaginary part.
        """ 
        identity = self.shortcut(x)
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out += identity
        return out

class ComplexResidualBlock(nn.Module):
    """
    A residual block for complex-valued data using the complexPyTorch library.
    This block assumes input and output channels are the same and does not perform downsampling.
    """
    def __init__(self, channels):
        super(ComplexResidualBlock, self).__init__()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = ComplexBatchNorm2d(channels)
        self.relu = CReLU()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = ComplexBatchNorm2d(channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A complex-valued input tensor.
        Returns:
            torch.Tensor: A complex-valued output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class Projection(nn.Module):
    """
    Modified projection for downsampling at the end of a stage, as described.
    It concatenates the input with a 1x1 convolution of the input, then subsamples.
    This effectively doubles the channel count and halves the spatial dimensions.
    """
    def __init__(self, in_channels):
        super(Projection, self).__init__()
        # 1x1 convolution with the same number of filters
        self.conv1x1 = ComplexConv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        # Subsampling by a factor of 2
        self.subsample = ComplexAvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A complex-valued input tensor from the previous stage.
        Returns:
            torch.Tensor: A complex-valued output tensor for the next stage.
        """
        # Apply 1x1 convolution
        conv_out = self.conv1x1(x)
        # Concatenate with the original output along the channel dimension
        concatenated_output = torch.cat([x, conv_out], dim=1)
        # Subsample the concatenated output
        return self.subsample(concatenated_output)

class ComplexResNet(nn.Module):
    """
    A complex-valued ResNet model implementing the described architecture.
    """
    def __init__(self, block, layers, input_channels=3, num_classes=10):
        super(ComplexResNet, self).__init__()
        
        # --- 1. Learn Initial Imaginary Component ---
        # This block is real-valued as it operates on the initial real input.
        self.learn_imaginary = LearnImagResidualBlock(input_channels, input_channels)

        # --- 2. Initial Operation on Complex Input ---
        # As described: Conv -> BN -> Activation on the newly formed complex input.
        self.initial_channels = 64
        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_channels, kernel_size=7, stride=2, padding=3, bias=True),
            ComplexBatchNorm2d(self.initial_channels),
            CReLU(),
            ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # --- 3. Residual Stages and Projections ---
        # Each stage consists of multiple residual blocks followed by a projection layer
        # to downsample and increase channel depth for the next stage.
        self.layer1 = self._make_layer(block, self.initial_channels, layers[0])
        self.proj1 = Projection(self.initial_channels) # 64 -> 128 channels

        self.layer2 = self._make_layer(block, 128, layers[1])
        self.proj2 = Projection(128) # 128 -> 256 channels
        
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.proj3 = Projection(256) # 256 -> 512 channels

        self.layer4 = self._make_layer(block, 512, layers[3])
        
        # --- 4. Classifier Head ---
        self.avgpool = ComplexAvgPool2d((1, 1))
        self.fc = ComplexLinear(512, num_classes)
        # add softmax here
    

    def _make_layer(self, block, channels, num_blocks):
        """Helper function to create a stage with multiple residual blocks."""
        layers = []
        for _ in range(num_blocks):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x_real):
        """
        Forward pass for the ComplexResNet.
        Args:
            x_real (torch.Tensor): A batch of real-valued input images.
        Returns:
            torch.Tensor: Real-valued logits for classification.
        """
        # Step 1: Learn the imaginary component from the real input
        x_imag = self.learn_imaginary(x_real)
        
        # Step 2: Create the complex tensor
        # Note: Input and learned imaginary part must have the same shape.
        x = torch.complex(x_real, x_imag)

        # Step 3: Pass through the initial complex operation
        x = self.initial_complex_op(x)

        # Step 4: Pass through the stages and projections
        x = self.layer1(x)
        x = self.proj1(x)
        x = self.layer2(x)
        x = self.proj2(x)
        x = self.layer3(x)
        x = self.proj3(x)
        x = self.layer4(x)
        
        # Step 5: Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # For classification, we take the magnitude of the complex output
        # to get real-valued logits.
        x = x.abs()

        return x

def ComplexResNet18(num_classes=10):
    """Constructor for a Complex ResNet-18 model."""
    return ComplexResNet(ComplexResidualBlock, [2, 2, 2, 2], num_classes=num_classes)

def ComplexResNet34(num_classes=10):
    """Constructor for a Complex ResNet-34 model."""
    return ComplexResNet(ComplexResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
