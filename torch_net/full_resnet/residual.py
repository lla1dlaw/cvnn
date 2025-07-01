"""
Author: Liam Laidlaw
Purpose: Classes for defining real and complex valued residual blocks.
Based on the architecture presented in "Deep Complex Networks", Trabelsi et al., 2018. 
View original paper here: https://openreview.net/forum?id=H1T2hmZAb

This version is portable and can dynamically build different ResNet architectures
('WS', 'DN', 'IB') and use different activation functions. It can also use
SyncBatchNorm for DDP or standard BatchNorm for single-process training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import  ComplexConv2d, ComplexLinear
from activations import CReLU, ZReLU, ModReLU, ComplexCardioid
# Import both standard and synchronized complex batch norm layers
from custom_complex_layers import ComplexBatchNorm2d, ComplexSyncBatchNorm2d

# MODULE: UTILITY & INITIALIZATION
# =================================

class ZeroImag(nn.Module):
    """Generates a zero tensor to be used as the imaginary part of a complex tensor."""
    def __init__(self):
        super(ZeroImag, self).__init__()

    def forward(self, x):
        """Generates a tensor of zeros with the same shape as the input."""
        return torch.zeros_like(x)

class ImaginaryComponentLearner(nn.Module):
    """
    A module to learn the imaginary component from a real-valued input, as
    described in the "Deep Complex Networks" paper (Sec 3.7).
    """
    def __init__(self, channels, use_sync_bn=False):
        super(ImaginaryComponentLearner, self).__init__()
        BN = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        self.layers = nn.Sequential(
            BN(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            BN(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.layers(x)

# MODULE: RESIDUAL BLOCKS
# ========================

class ComplexResidualBlock(nn.Module):
    """A residual block for complex-valued data that can use SyncBatchNorm."""
    def __init__(self, channels, activation_fn_class, use_sync_bn=False):
        super(ComplexResidualBlock, self).__init__()
        ComplexBN = ComplexSyncBatchNorm2d if use_sync_bn else ComplexBatchNorm2d
        self.bn1 = ComplexBN(channels)
        self.relu1 = activation_fn_class()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = ComplexBN(channels)
        self.relu2 = activation_fn_class()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
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
    """A real-valued residual block that can use SyncBatchNorm."""
    def __init__(self, channels, use_sync_bn=False):
        super(RealResidualBlock, self).__init__()
        BN = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        self.bn1 = BN(channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BN(channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
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
    """A DDP-compliant complex-valued ResNet model that can be configured with different architectures."""
    def __init__(self, block_class, architecture_type, activation_function, learn_imaginary_component, use_sync_bn=False, input_channels=3, num_classes=10):
        super(ComplexResNet, self).__init__()
        
        configs = {
            'WS': {'filters': 12, 'blocks_per_stage': [16, 16, 16]},
            'DN': {'filters': 10, 'blocks_per_stage': [23, 23, 23]},
            'IB': {'filters': 11, 'blocks_per_stage': [19, 19, 19]}
        }
        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        
        activation_map = {'crelu': CReLU, 'zrelu': ZReLU, 'modrelu': ModReLU, 'complex_cardioid': ComplexCardioid}
        # **FIX**: Store the activation function CLASS, not an instance.
        self.activation_fn_class = activation_map.get(activation_function.lower())
        if self.activation_fn_class is None:
            raise ValueError(f"Unknown activation function: {activation_function}")
        
        ComplexBN = ComplexSyncBatchNorm2d if use_sync_bn else ComplexBatchNorm2d

        if learn_imaginary_component:
            self.imag_handler = ImaginaryComponentLearner(input_channels, use_sync_bn=use_sync_bn) 
        else:
            self.imag_handler = ZeroImag()

        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBN(self.initial_filters),
            # **FIX**: Instantiate the activation function here.
            self.activation_fn_class()
        )
        
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks, use_sync_bn))
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(
                    ComplexConv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False)
                )
            current_channels *= 2

        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ComplexLinear(final_channels, num_classes)

    def _make_stage(self, block_class, channels, num_blocks, use_sync_bn):
        blocks = []
        for _ in range(num_blocks):
            # **FIX**: Pass the activation CLASS to the block constructor.
            blocks.append(block_class(channels, self.activation_fn_class, use_sync_bn=use_sync_bn))
        return nn.Sequential(*blocks)

    def forward(self, x_real):
        x_imag = self.imag_handler(x_real)
        x = torch.complex(x_real, x_imag)
        x = self.initial_complex_op(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                projection_conv = self.downsample_layers[i]
                projected_x = projection_conv(x)
                x = torch.cat([x, projected_x], dim=1)
                pooled_real = F.avg_pool2d(x.real, kernel_size=2, stride=2)
                pooled_imag = F.avg_pool2d(x.imag, kernel_size=2, stride=2)
                x = torch.complex(pooled_real, pooled_imag)
        
        pooled_real = self.avgpool(x.real)
        pooled_imag = self.avgpool(x.imag)
        x = torch.complex(pooled_real, pooled_imag)
        x = torch.flatten(x, 1)
        x_complex_logits = self.fc(x)
        return x_complex_logits.abs()


class RealResNet(nn.Module):
    """A DDP-compliant real-valued ResNet model that can be configured with different architectures."""
    def __init__(self, block_class, architecture_type, use_sync_bn=False, input_channels=3, num_classes=10):
        super(RealResNet, self).__init__()
        
        configs = {
            'WS': {'filters': 18, 'blocks_per_stage': [14, 14, 14]},
            'DN': {'filters': 14, 'blocks_per_stage': [23, 23, 23]},
            'IB': {'filters': 16, 'blocks_per_stage': [18, 18, 18]}
        }
        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        
        BN = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        
        self.initial_op = nn.Sequential(
            nn.Conv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            BN(self.initial_filters),
            nn.ReLU(inplace=False)
        )
        
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks, use_sync_bn))
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False)
                )
            current_channels *= 2
        
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_stage(self, block_class, channels, num_blocks, use_sync_bn):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels, use_sync_bn=use_sync_bn))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.initial_op(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                projection_conv = self.downsample_layers[i]
                projected_x = projection_conv(x)
                x = torch.cat([x, projected_x], dim=1)
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
