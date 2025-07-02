"""
Author: Liam Laidlaw
Purpose: Classes for defining real and complex valued residual blocks.
Based on the architecture presented in "Deep Complex Networks", Trabelsi et al., 2018. 
View original paper here: https://openreview.net/forum?id=H1T2hmZAb

This version is configured for use with torch.nn.DataParallel and includes
the specific weight initialization from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
from complexPyTorch.complexLayers import  ComplexConv2d, ComplexLinear
from activations import CReLU, ZReLU, ModReLU, ComplexCardioid
# Import the DataParallel-compatible complex batch norm layer
from custom_complex_layers import ComplexBatchNorm2d
import math

# --- WEIGHT INITIALIZATION (as per Paper Sec. 3.6) ---

def complex_rayleigh_init(weight_tensor, fan_in):
    """Initializes complex weights using Rayleigh distribution for magnitude
    and uniform for phase, scaled by He variance."""
    # Calculate sigma for Rayleigh distribution based on He initialization
    # Var(W) = 2 * sigma^2 = 2 / fan_in  =>  sigma = 1 / sqrt(fan_in)
    sigma = 1 / math.sqrt(fan_in)
    
    # Magnitude from Rayleigh distribution
    magnitude = torch.randn_like(weight_tensor.real) * sigma
    
    # Phase from Uniform distribution
    phase = torch.rand_like(weight_tensor.real) * (2 * math.pi) - math.pi
    
    # Create complex tensor
    return torch.polar(magnitude, phase)

def unitary_init(weight_tensor, fan_in):
    """
    Performs unitary initialization and scales to match He variance.
    This is the variant used in the paper's experiments.
    """
    # Create a random complex matrix
    num_rows, num_cols = weight_tensor.shape[0], weight_tensor.shape[1]
    
    # Flatten the spatial dimensions of the kernel
    flat_shape = (num_rows, num_cols * weight_tensor.shape[2] * weight_tensor.shape[3])
    
    # Generate a random complex matrix
    random_matrix = torch.randn(flat_shape, dtype=torch.complex64)
    
    # Perform QR decomposition to get a unitary matrix Q
    q, _ = torch.linalg.qr(random_matrix)
    
    # Reshape back to the original weight tensor shape
    unitary_matrix = q.reshape(weight_tensor.shape)
    
    # --- Scaling to match He variance ---
    # He variance for complex is 2 / fan_in
    he_variance = 2.0 / fan_in
    
    # Variance of the unitary matrix (should be close to 1/num_rows)
    # We scale to match the desired He variance
    scaling_factor = math.sqrt(he_variance)
    
    with torch.no_grad():
        weight_tensor.copy_(unitary_matrix * scaling_factor)
    return weight_tensor


def init_weights(m):
    """
    Applies the paper's weight initialization to the model's layers.
    """
    if isinstance(m, nn.Conv2d):
        # Orthogonal initialization for real-valued convolutions
        fan_in = nn.init._calculate_fan_in_and_fan_out(m.weight)[0]
        
        # Create a random tensor to be orthogonalized
        flat_shape = (m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        random_matrix = torch.randn(flat_shape)
        
        # Perform orthogonal initialization
        orthogonal_matrix = orthogonal_(random_matrix)
        
        # Reshape back to the original weight tensor shape
        reshaped_matrix = orthogonal_matrix.reshape(m.weight.shape)
        
        # --- Scaling to match He variance ---
        # He variance is 2 / fan_in
        he_variance = 2.0 / fan_in
        
        # Variance of an orthogonal matrix is 1/N
        # We scale to match the desired He variance
        scaling_factor = math.sqrt(he_variance * m.out_channels)
        
        with torch.no_grad():
            m.weight.copy_(reshaped_matrix * scaling_factor)

    elif isinstance(m, ComplexConv2d):
        # Unitary initialization for complex-valued convolutions
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        unitary_init(m.weight, fan_in)


# MODULE: UTILITY & INITIALIZATION
# =================================

class ZeroImag(nn.Module):
    def __init__(self):
        super(ZeroImag, self).__init__()
    def forward(self, x):
        return torch.zeros_like(x)

class ImaginaryComponentLearner(nn.Module):
    def __init__(self, channels):
        super(ImaginaryComponentLearner, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x):
        return self.layers(x)

# MODULE: RESIDUAL BLOCKS
# ========================

class ComplexResidualBlock(nn.Module):
    def __init__(self, channels, activation_fn_class):
        super(ComplexResidualBlock, self).__init__()
        self.bn1 = ComplexBatchNorm2d(channels)
        self.relu1 = activation_fn_class()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(channels)
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
    def __init__(self, channels):
        super(RealResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
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
    def __init__(self, block_class, architecture_type, activation_function, learn_imaginary_component, input_channels=3, num_classes=10):
        super(ComplexResNet, self).__init__()
        configs = {'WS': {'filters': 12, 'blocks_per_stage': [16, 16, 16]}, 'DN': {'filters': 10, 'blocks_per_stage': [23, 23, 23]}, 'IB': {'filters': 11, 'blocks_per_stage': [19, 19, 19]}}
        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        activation_map = {'crelu': CReLU, 'zrelu': ZReLU, 'modrelu': ModReLU, 'complex_cardioid': ComplexCardioid}
        self.activation_fn_class = activation_map.get(activation_function.lower())
        if self.activation_fn_class is None:
            raise ValueError(f"Unknown activation function: {activation_function}")
        if learn_imaginary_component:
            self.imag_handler = ImaginaryComponentLearner(input_channels) 
        else:
            self.imag_handler = ZeroImag()
        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(self.initial_filters),
            self.activation_fn_class()
        )
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks))
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(ComplexConv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False))
            current_channels *= 2
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ComplexLinear(final_channels, num_classes)
        self.apply(init_weights)
    def _make_stage(self, block_class, channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels, self.activation_fn_class))
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
    def __init__(self, block_class, architecture_type, input_channels=3, num_classes=10):
        super(RealResNet, self).__init__()
        configs = {'WS': {'filters': 18, 'blocks_per_stage': [14, 14, 14]}, 'DN': {'filters': 14, 'blocks_per_stage': [23, 23, 23]}, 'IB': {'filters': 16, 'blocks_per_stage': [18, 18, 18]}}
        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        self.initial_op = nn.Sequential(
            nn.Conv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_filters),
            nn.ReLU(inplace=False)
        )
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks))
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False))
            current_channels *= 2
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)
        self.apply(init_weights)
    def _make_stage(self, block_class, channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels))
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
