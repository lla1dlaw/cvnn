"""
Custom Complex-Valued Layers for PyTorch

This module provides DDP-compliant and standard implementations of complex
batch normalization by wrapping the native PyTorch layers.
"""
import torch
import torch.nn as nn

class ComplexBatchNorm2d(nn.Module):
    """
    A standard complex batch normalization layer for single-GPU or CPU training.

    This layer wraps the native torch.nn.BatchNorm2d. It works by splitting
    the complex input into real and imaginary parts, applying BatchNorm2d to the
    concatenated real-valued tensor, and then recombining the results.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(
            num_features * 2,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x):
        if not x.is_complex():
            raise TypeError("Input must be a complex tensor.")
        
        concatenated_input = torch.cat([x.real, x.imag], dim=1)
        bn_output = self.bn(concatenated_input)
        real_part, imag_part = torch.split(bn_output, self.num_features, dim=1)
        
        return torch.complex(real_part, imag_part)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.num_features}, '
                f'eps={self.bn.eps}, momentum={self.bn.momentum}, '
                f'affine={self.bn.affine}, '
                f'track_running_stats={self.bn.track_running_stats})')


class ComplexSyncBatchNorm2d(nn.Module):
    """
    A DistributedDataParallel-compliant complex batch normalization layer.

    This layer wraps the native torch.nn.SyncBatchNorm, which correctly
    synchronizes statistics (mean and variance) across multiple GPUs.
    It works by splitting the complex input into real and imaginary parts,
    applying SyncBatchNorm to the concatenated real-valued tensor, and then
    recombining the results.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexSyncBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.sync_bn = nn.SyncBatchNorm(
            num_features * 2,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x):
        if not x.is_complex():
            raise TypeError("Input must be a complex tensor.")
        
        concatenated_input = torch.cat([x.real, x.imag], dim=1)
        bn_output = self.sync_bn(concatenated_input)
        real_part, imag_part = torch.split(bn_output, self.num_features, dim=1)
        
        return torch.complex(real_part, imag_part)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.num_features}, '
                f'eps={self.sync_bn.eps}, momentum={self.sync_bn.momentum}, '
                f'affine={self.sync_bn.affine}, '
                f'track_running_stats={self.sync_bn.track_running_stats})')
