"""
Author: Liam Laidlaw
Purpose: Classes for defining real and complex valued residual blocks.
Based on the architecture presented in "Deep Complex Networks", Trabelsi et al., 2018. 
View original paper here: https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
import torch.nn as nn
from complexPyTorch.complexLayers import  ComplexConv2d, ComplexLinear
from activations import ModReLU, ZReLU, CReLU, ComplexCardioid # Assuming activations.py is in the same directory
from custom_complex_layers import ComplexBatchNorm2d

# MODULE: UTILITY & INITIALIZATION
# =================================

class ZeroImag(nn.Module):
    """Transforms a real-valued tensor into a complex-valued one by setting the
    imaginary part to zero.

    This module is used when the imaginary component of the input is not
    explicitly learned.
    """
    def __init__(self):
        super(ZeroImag, self).__init__()

    def forward(self, x):
        """Casts the real-valued input to a complex tensor.

        Args:
            x (torch.Tensor): The real-valued input tensor.

        Returns:
            torch.Tensor: The complex-valued output tensor with a zero imaginary part.
        """
        return x.to(torch.complex64)

class LearnImag(nn.Module):
    """A real-valued module that learns an imaginary component from a real input.

    As per the paper "Deep Complex Networks" (Sec 3.7), an initial imaginary
    component can be learned from the real-valued input data. This module
    uses a simple Conv-BN-ReLU sequence to generate this component.
    """
    def __init__(self, in_channels):
        """Initializes the LearnImag module.

        Args:
            in_channels (int): The number of channels in the input tensor.
        """
        super(LearnImag, self).__init__()
        self.imag_learner = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Generates a learned imaginary component.

        Args:
            x (torch.Tensor): The real-valued input tensor.

        Returns:
            torch.Tensor: A real-valued tensor representing the learned imaginary part.
        """
        return self.imag_learner(x)


# MODULE: RESIDUAL BLOCKS
# ========================

class ComplexResidualBlock(nn.Module):
    """A residual block for complex-valued data based on the "pre-activation" design.

    This block follows the architecture shown in Figure 1b (left) of the
    "Deep Complex Networks" paper.
    The structure is: Complex BN -> Activation -> Complex Conv -> Complex BN -> Activation -> Complex Conv.
    The identity shortcut is added to the output of the second convolution.

    Attributes:
        bn1 (ComplexBatchNorm2d): The first complex batch normalization layer.
        relu1 (nn.Module): The first activation function.
        conv1 (ComplexConv2d): The first complex convolutional layer.
        bn2 (ComplexBatchNorm2d): The second complex batch normalization layer.
        relu2 (nn.Module): The second activation function.
        conv2 (ComplexConv2d): The second complex convolutional layer.
    """
    def __init__(self, channels, activation_fn):
        """Initializes the ComplexResidualBlock.

        Args:
            channels (int): The number of input and output channels.
            activation_fn (nn.Module): The class of the activation function to use (e.g., CReLU).
        """
        super(ComplexResidualBlock, self).__init__()
        self.bn1 = ComplexBatchNorm2d(channels)
        self.relu1 = activation_fn()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = ComplexBatchNorm2d(channels)
        self.relu2 = activation_fn()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """Performs the forward pass through the complex residual block.

        Args:
            x (torch.Tensor): A complex-valued input tensor.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the block.
        """
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
    """A real-valued residual block designed for fair comparison with the complex version.

    This block follows the same "pre-activation" style as its complex counterpart.
    The structure is: BN -> ReLU -> Conv -> BN -> ReLU -> Conv.
    """
    def __init__(self, channels):
        """Initializes the RealResidualBlock.

        Args:
            channels (int): The number of input and output channels.
        """
        super(RealResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """Performs the forward pass through the real residual block.

        Args:
            x (torch.Tensor): A real-valued input tensor.

        Returns:
            torch.Tensor: The real-valued output tensor after applying the block.
        """
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
    """A complex-valued ResNet model based on "Deep Complex Networks".

    This class implements the residual network architecture described in the paper,
    including logic for handling complex inputs, residual stages, and a classifier head.
    """
    def __init__(self, block_class, activation_function, architecture_type, learn_imaginary_component, input_channels=3, num_classes=10):
        """Initializes the ComplexResNet model.

        Args:
            block_class (nn.Module): The class for the residual block (e.g., ComplexResidualBlock).
            activation_function (str): The name of the complex activation to use ('crelu', 'modrelu', 'zrelu').
            architecture_type (str): The type of architecture ('WS', 'DN', 'IB') which defines
                the filter counts and block depths.
            learn_imaginary_component (bool): If True, learns the initial imaginary part
                from the input; otherwise, it is zero-initialized.
            input_channels (int, optional): The number of channels in the input image. Defaults to 3.
            num_classes (int, optional): The number of output classes for the classifier. Defaults to 10.
        
        Raises:
            ValueError: If an unknown activation function name is provided.
        """
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
        
        activation_map = {'crelu': CReLU, 'modrelu': ModReLU, 'zrelu': ZReLU, 'complex_cardioid': ComplexCardioid}
        self.activation_fn = activation_map.get(activation_function.lower())
        if self.activation_fn is None:
            raise ValueError(f"Unknown activation function: {activation_function}")

        # --- Layer Definitions ---
        self.imag_handler = LearnImag(input_channels) if learn_imaginary_component else ZeroImag()
        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(self.initial_filters),
            self.activation_fn()
        )
        
        self.stages = nn.ModuleList()
        in_channels = self.initial_filters
        for num_blocks in self.blocks_per_stage:
            self.stages.append(self._make_stage(block_class, in_channels, num_blocks))
            in_channels *= 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ComplexLinear(self.initial_filters * (2**(len(self.blocks_per_stage)-1)), num_classes)

    def _make_stage(self, block_class, channels, num_blocks):
        """Creates a single stage of the ResNet.

        A stage consists of one or more residual blocks that operate at the same
        feature map resolution.

        Args:
            block_class (nn.Module): The class of the residual block to use.
            channels (int): The number of channels for the blocks in this stage.
            num_blocks (int): The number of residual blocks in this stage.

        Returns:
            nn.Sequential: A sequential container of the residual blocks for the stage.
        """
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels, self.activation_fn))
        return nn.Sequential(*blocks)

    def _make_projection(self, in_channels, out_channels):
        """Creates the projection layer for downsampling between stages.

        This layer is responsible for halving the spatial dimensions and doubling
        the number of channels.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns:
            nn.Sequential: The projection layer.
        """
        return nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            ComplexBatchNorm2d(out_channels)
        )

    def forward(self, x_real):
        """Performs the forward pass for the ComplexResNet.

        Args:
            x_real (torch.Tensor): A batch of real-valued input images.

        Returns:
            torch.Tensor: Real-valued logits for classification.
        """
        x_imag = self.imag_handler(x_real)
        x = torch.complex(x_real, x_imag)
        x = self.initial_complex_op(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                in_ch = x.shape[1]
                out_ch = in_ch * 2
                projection = self._make_projection(in_ch, out_ch).to(x.device)
                x = projection(x)
        
        x = self.avgpool(x.abs())
        x = torch.flatten(x, 1)
        x_complex_logits = self.fc(x)

        return x_complex_logits.abs()


class RealResNet(nn.Module):
    """A real-valued ResNet designed for fair comparison with the ComplexResNet.

    This model mirrors the architecture of the ComplexResNet but uses real-valued
    operations. It doubles the number of filters at each stage to have a
    parameter count that is comparable to its complex counterpart.
    """
    def __init__(self, block_class, architecture_type, input_channels=3, num_classes=10):
        """Initializes the RealResNet model.

        Args:
            block_class (nn.Module): The class for the residual block (e.g., RealResidualBlock).
            architecture_type (str): The type of architecture ('WS', 'DN', 'IB').
            input_channels (int, optional): The number of channels in the input image. Defaults to 3.
            num_classes (int, optional): The number of output classes. Defaults to 10.
        """
        super(RealResNet, self).__init__()

        configs = {
            'WS': {'filters': 16, 'blocks_per_stage': [2, 2, 2]},
            'DN': {'filters': 12, 'blocks_per_stage': [4, 4, 4]},
            'IB': {'filters': 14, 'blocks_per_stage': [3, 3, 3]}
        }
        config = configs[architecture_type]
        self.initial_filters = config['filters'] * 2
        self.blocks_per_stage = config['blocks_per_stage']
        
        self.initial_op = nn.Sequential(
            nn.Conv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_filters),
            nn.ReLU(inplace=True)
        )
        
        self.stages = nn.ModuleList()
        in_channels = self.initial_filters
        for num_blocks in self.blocks_per_stage:
            self.stages.append(self._make_stage(block_class, in_channels, num_blocks))
            in_channels *= 2
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.initial_filters * (2**(len(self.blocks_per_stage)-1)), num_classes)

    def _make_stage(self, block_class, channels, num_blocks):
        """Creates a single stage of the real-valued ResNet.

        Args:
            block_class (nn.Module): The class of the residual block to use.
            channels (int): The number of channels for the blocks in this stage.
            num_blocks (int): The number of residual blocks in this stage.

        Returns:
            nn.Sequential: A sequential container of the residual blocks for the stage.
        """
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels))
        return nn.Sequential(*blocks)
    
    def _make_projection(self, in_channels, out_channels):
        """Creates the real-valued projection layer for downsampling.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns:
            nn.Sequential: The projection layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """Performs the forward pass for the RealResNet.

        Args:
            x (torch.Tensor): A batch of real-valued input images.

        Returns:
            torch.Tensor: Real-valued logits for classification.
        """
        x = self.initial_op(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                in_ch = x.shape[1]
                out_ch = in_ch * 2
                projection = self._make_projection(in_ch, out_ch).to(x.device)
                x = projection(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
