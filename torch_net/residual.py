"""
Author: Liam Laidlaw
Purpose: Classes for defining real and complex valued residual blocks.
Based on the architecture presented in "Deep Complex Networks", Trabelsi et al., 2018. 
View original paper here: https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import  ComplexConv2d, ComplexLinear
from activations import CReLU # Assuming activations.py is in the same directory
from custom_complex_layers import ComplexBatchNorm2d

# MODULE: UTILITY & INITIALIZATION
# =================================

class ZeroImag(nn.Module):
    """Generates a zero tensor to be used as the imaginary part of a complex tensor."""
    def __init__(self):
        super(ZeroImag, self).__init__()

    def forward(self, x):
        """Generates a tensor of zeros with the same shape as the input."""
        return torch.zeros_like(x)

# MODULE: RESIDUAL BLOCKS
# ========================

class ComplexResidualBlock(nn.Module):
    """A residual block for complex-valued data based on the "pre-activation" design."""
    def __init__(self, channels, activation_fn):
        """Initializes the ComplexResidualBlock."""
        super(ComplexResidualBlock, self).__init__()
        self.bn1 = ComplexBatchNorm2d(channels)
        self.relu1 = activation_fn()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = ComplexBatchNorm2d(channels)
        self.relu2 = activation_fn()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """Performs the forward pass through the complex residual block."""
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = out + identity
        return out

class RealResidualBlock(nn.Module):
    """A real-valued residual block designed for fair comparison with the complex version."""
    def __init__(self, channels):
        """Initializes the RealResidualBlock."""
        super(RealResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """Performs the forward pass through the real residual block."""
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = out + identity
        return out


# MODULE: NETWORK ARCHITECTURES
# ==============================

class ComplexResNet(nn.Module):
    """
    A complex-valued ResNet model strictly following the architecture from
    "Deep Complex Networks" (Trabelsi et al., 2018).
    """
    def __init__(self, block_class, activation_function, architecture_type, learn_imaginary_component, input_channels=3, num_classes=10):
        """Initializes the ComplexResNet model."""
        super(ComplexResNet, self).__init__()
        
        # --- Architecture Configuration ---
        configs = {
            'WS': {'filters': 16, 'blocks_per_stage': [2, 2, 2]},
            'DN': {'filters': 12, 'blocks_per_stage': [4, 4, 4]},
            'IB': {'filters': 14, 'blocks_per_stage': [3, 3, 3]}
        }
        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        
        # Using CReLU as it was found to be most effective in the paper's experiments.
        self.activation_fn = CReLU

        # --- Layer Definitions ---
        # 1. Initial Imaginary Component Generation (as per paper Sec 3.7)
        if learn_imaginary_component:
            self.imag_handler = RealResidualBlock(input_channels) 
        else:
            self.imag_handler = ZeroImag()

        # 2. Initial Complex Operation
        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(self.initial_filters),
            self.activation_fn()
        )
        
        # 3. Residual Stages
        self.stages = nn.ModuleList()
        current_channels = self.initial_filters
        for num_blocks in self.blocks_per_stage:
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks))
            current_channels *= 2

        # 4. Classifier Head
        # The final number of channels after all stages and downsampling
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ComplexLinear(final_channels, num_classes)

    def _make_stage(self, block_class, channels, num_blocks):
        """Creates a single stage of the ResNet."""
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels, self.activation_fn))
        return nn.Sequential(*blocks)

    def forward(self, x_real):
        """Performs the forward pass for the ComplexResNet."""
        x_imag = self.imag_handler(x_real)
        x = torch.complex(x_real, x_imag)
        x = self.initial_complex_op(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            # Apply downsampling block after each stage except the last one
            if i < len(self.stages) - 1:
                # Paper's downsampling method (Sec 3.7)
                # 1. Apply a 1x1 convolution with the same number of filters
                projection_conv = ComplexConv2d(
                    in_channels=x.shape[1], 
                    out_channels=x.shape[1],
                    kernel_size=1, 
                    stride=1,
                    bias=False
                ).to(x.device)
                projected_x = projection_conv(x)
                
                # 2. Concatenate along channel dimension (doubles channels)
                x = torch.cat([x, projected_x], dim=1)
                
                # 3. Subsample using average pooling (halves spatial dimensions)
                # **FIX 1: Manually perform pooling on real and imaginary parts**
                pooled_real = F.avg_pool2d(x.real, kernel_size=2, stride=2)
                pooled_imag = F.avg_pool2d(x.imag, kernel_size=2, stride=2)
                x = torch.complex(pooled_real, pooled_imag)
        
        # Classifier Head
        # **FIX 2: Manually perform adaptive pooling on real and imaginary parts**
        pooled_real = self.avgpool(x.real)
        pooled_imag = self.avgpool(x.imag)
        x = torch.complex(pooled_real, pooled_imag)

        x = torch.flatten(x, 1)
        x_complex_logits = self.fc(x)

        # Return the magnitude of the final complex logits for the loss function
        return x_complex_logits.abs()


class RealResNet(nn.Module):
    """
    A real-valued ResNet with a mirrored architecture for fair comparison.
    """
    def __init__(self, block_class, architecture_type, input_channels=3, num_classes=10):
        """Initializes the RealResNet model."""
        super(RealResNet, self).__init__()

        configs = {
            'WS': {'filters': 16, 'blocks_per_stage': [2, 2, 2]},
            'DN': {'filters': 12, 'blocks_per_stage': [4, 4, 4]},
            'IB': {'filters': 14, 'blocks_per_stage': [3, 3, 3]}
        }
        config = configs[architecture_type]
        # Double the filters to have a comparable number of parameters to the complex model
        self.initial_filters = config['filters'] * 2
        self.blocks_per_stage = config['blocks_per_stage']
        
        self.initial_op = nn.Sequential(
            nn.Conv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_filters),
            nn.ReLU(inplace=True)
        )
        
        self.stages = nn.ModuleList()
        current_channels = self.initial_filters
        for num_blocks in self.blocks_per_stage:
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks))
            current_channels *= 2
        
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_stage(self, block_class, channels, num_blocks):
        """Creates a single stage of the real-valued ResNet."""
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        """Performs the forward pass for the RealResNet."""
        x = self.initial_op(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # **FIX 3: Corrected loop condition**
            if i < len(self.stages) - 1:
                # Mirror the complex network's downsampling logic for a fair comparison.
                # 1. Apply a 1x1 convolution.
                projection_conv = nn.Conv2d(
                    in_channels=x.shape[1], 
                    out_channels=x.shape[1], # Same number of filters
                    kernel_size=1, 
                    stride=1,
                    bias=False
                ).to(x.device)
                projected_x = projection_conv(x)
                
                # 2. Concatenate along the channel dimension to double the channels.
                x = torch.cat([x, projected_x], dim=1)
                
                # 3. Subsample using average pooling to halve spatial dimensions.
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
