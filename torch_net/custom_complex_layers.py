"""
Custom Complex-Valued Layers for PyTorch

This module provides a custom implementation of ComplexBatchNorm2d that is
compatible with torch.nn.DataParallel for multi-GPU training.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter

class ComplexBatchNorm2d(nn.Module):
    """
    A DataParallel-compatible implementation of the complex batch normalization
    described in "Deep Complex Networks" (Trabelsi et al., 2018).

    This layer performs a 2D whitening operation that decorrelates the real
    and imaginary parts of the complex-valued activations.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True):
        """
        Initializes the ComplexBatchNorm2d layer.

        Args:
            num_features (int): The number of complex-valued channels.
            eps (float): A small value added for numerical stability.
            momentum (float): The momentum for updating running statistics.
            track_running_stats (bool): If True, tracks running mean and covariance.
        """
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # Learnable affine parameters: a complex bias (beta) and a 2x2 scaling matrix (gamma)
        # gamma is represented by 3 values for the symmetric 2x2 matrix: (rr, ii, ri)
        self.weight = Parameter(torch.Tensor(num_features, 3)) 
        # beta is represented by 2 values for the complex number: (r, i)
        self.bias = Parameter(torch.Tensor(num_features, 2))   

        # Running statistics are stored as real tensors
        if self.track_running_stats:
            # running_mean stores real and imaginary parts separately
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            # running_cov stores V_rr, V_ii, V_ri
            self.register_buffer('running_cov', torch.zeros(num_features, 3))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_cov', None)
            self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the parameters and running statistics."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_cov.zero_()
            # Initialize covariance to identity matrix (V_rr=1, V_ii=1, V_ri=0)
            self.running_cov[:, 0] = 1 
            self.running_cov[:, 1] = 1 
            self.num_batches_tracked.zero_()
        
        # Initialize bias (beta) to zero
        self.bias.data.zero_()
        
        # Initialize weight (gamma) to identity matrix (gamma_rr=1, gamma_ii=1, gamma_ri=0)
        self.weight.data.zero_()
        self.weight.data[:, 0] = 1.0 
        self.weight.data[:, 1] = 1.0

    def forward(self, x):
        """Performs the forward pass."""
        if not x.is_complex():
            raise TypeError("Input must be a complex tensor.")

        # --- 1. Calculate or Retrieve Statistics ---
        if self.training and self.track_running_stats:
            # --- Training Mode: Calculate batch statistics and update running stats ---
            
            # Calculate batch mean (complex tensor of shape [num_features])
            mean_complex = x.mean(dim=[0, 2, 3])
            
            # **FIX 1: Shape Mismatch**
            # Convert complex mean to a real tensor of shape [num_features, 2] for update
            mean_for_update = torch.stack([mean_complex.real, mean_complex.imag], dim=1).detach()

            # Update running mean
            self.num_batches_tracked.add_(1)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean_for_update

            # Center the data using the calculated batch mean
            centered_x = x - mean_complex.view(1, self.num_features, 1, 1)

            # Calculate covariance matrix elements
            V_rr = (centered_x.real ** 2).mean(dim=[0, 2, 3])
            V_ii = (centered_x.imag ** 2).mean(dim=[0, 2, 3])
            V_ri = (centered_x.real * centered_x.imag).mean(dim=[0, 2, 3])
            
            # Stack covariance elements for update and detach
            cov_for_update = torch.stack([V_rr, V_ii, V_ri], dim=1).detach()

            # Update running covariance
            self.running_cov.data = (1 - self.momentum) * self.running_cov.data + self.momentum * cov_for_update
            
            # Use batch statistics for normalization during training
            mean_to_use = mean_complex
            cov_to_use = torch.stack([V_rr, V_ii, V_ri], dim=1)

        else:
            # --- Evaluation Mode: Use running statistics ---
            mean_to_use = torch.complex(self.running_mean[:, 0], self.running_mean[:, 1])
            cov_to_use = self.running_cov
        
        # --- 2. Whiten the Data ---
        # Reshape mean for broadcasting
        mean_reshaped = mean_to_use.view(1, self.num_features, 1, 1)
        centered_x = x - mean_reshaped
        
        # Extract covariance components and reshape for broadcasting
        V_rr = cov_to_use[:, 0].view(1, self.num_features, 1, 1)
        V_ii = cov_to_use[:, 1].view(1, self.num_features, 1, 1)
        V_ri = cov_to_use[:, 2].view(1, self.num_features, 1, 1)
        
        # Calculate the inverse square root of the covariance matrix
        # s = determinant of the covariance matrix
        s = V_rr * V_ii - V_ri ** 2
        # t = sqrt of the determinant
        t = torch.sqrt(s + self.eps)
        # inv_t = 1 / t
        inv_t = 1.0 / (t + self.eps)
        
        # **FIX 2: Mathematical Error**
        # The whitening matrix components are functions of t (sqrt of det), not s (det).
        Rrr = (V_ii + t) * inv_t
        Rii = (V_rr + t) * inv_t
        Rri = -V_ri * inv_t

        # Apply the whitening transformation
        real_part = Rrr * centered_x.real + Rri * centered_x.imag
        imag_part = Rri * centered_x.real + Rii * centered_x.imag
        whitened_x = torch.complex(real_part, imag_part)

        # --- 3. Apply Affine Transformation (gamma * x_hat + beta) ---
        # Reshape gamma and beta for broadcasting
        gamma_rr = self.weight[:, 0].view(1, self.num_features, 1, 1)
        gamma_ii = self.weight[:, 1].view(1, self.num_features, 1, 1)
        gamma_ri = self.weight[:, 2].view(1, self.num_features, 1, 1)
        beta_r = self.bias[:, 0].view(1, self.num_features, 1, 1)
        beta_i = self.bias[:, 1].view(1, self.num_features, 1, 1)

        # Apply affine transformation
        out_real = gamma_rr * whitened_x.real + gamma_ri * whitened_x.imag + beta_r
        out_imag = gamma_ri * whitened_x.real + gamma_ii * whitened_x.imag + beta_i

        return torch.complex(out_real, out_imag)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.num_features}, '
                f'eps={self.eps}, momentum={self.momentum}, '
                f'track_running_stats={self.track_running_stats})')
