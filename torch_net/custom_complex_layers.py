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
        self.weight = Parameter(torch.Tensor(num_features, 3)) # For gamma: (rr, ii, ri)
        self.bias = Parameter(torch.Tensor(num_features, 2))   # For beta: (r, i)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            # running_cov stores V_rr, V_ii, V_ri
            self.register_buffer('running_cov', torch.zeros(num_features, 3))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_cov', None)
            self.register_parameter('num_batches_tracked', None)
            
        self.reset_parameters()

    def reset_running_stats(self):
        """Resets the running statistics to their initial values."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_cov.zero_()
            # Initialize covariance to identity matrix
            self.running_cov[:, 0] = 1.0
            self.running_cov[:, 1] = 1.0
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        """Initializes the learnable parameters and running statistics."""
        self.reset_running_stats()
        # Initialize gamma to be an identity matrix
        self.weight.data[:, 0].fill_(1)
        self.weight.data[:, 1].fill_(1)
        self.weight.data[:, 2].zero_()
        # Initialize beta to zero
        self.bias.data.zero_()

    def forward(self, x):
        """
        Performs the forward pass.
        """
        # Ensure input is complex
        if not x.is_complex():
            raise ValueError("Input to ComplexBatchNorm2d must be complex.")

        # Get the batch and spatial dimensions for calculations
        batch_size, _, height, width = x.shape
        num_elements = batch_size * height * width

        # --- 1. Calculate Statistics ---
        if self.training or not self.track_running_stats:
            # Calculate mean across batch and spatial dimensions
            # (B, C, H, W) -> (C)
            mean = torch.mean(x, dim=[0, 2, 3])
            
            # Center the input
            centered_x = x - mean.view(1, self.num_features, 1, 1)

            # Calculate covariance matrix components V
            # V_rr = Var(real)
            V_rr = torch.sum(centered_x.real ** 2, dim=[0, 2, 3]) / num_elements
            # V_ii = Var(imag)
            V_ii = torch.sum(centered_x.imag ** 2, dim=[0, 2, 3]) / num_elements
            # V_ri = Cov(real, imag)
            V_ri = torch.sum(centered_x.real * centered_x.imag, dim=[0, 2, 3]) / num_elements

            # Update running stats during training
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                # Update running mean
                self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.detach()
                # Update running covariance (with unbiased estimator correction)
                unbiased_factor = num_elements / (num_elements - 1) if num_elements > 1 else 1.0
                self.running_cov.data[:, 0] = (1 - self.momentum) * self.running_cov.data[:, 0] + self.momentum * V_rr.detach() * unbiased_factor
                self.running_cov.data[:, 1] = (1 - self.momentum) * self.running_cov.data[:, 1] + self.momentum * V_ii.detach() * unbiased_factor
                self.running_cov.data[:, 2] = (1 - self.momentum) * self.running_cov.data[:, 2] + self.momentum * V_ri.detach() * unbiased_factor

        else: # During evaluation, use the running stats
            mean = self.running_mean
            centered_x = x - mean.view(1, self.num_features, 1, 1)
            V_rr = self.running_cov[:, 0]
            V_ii = self.running_cov[:, 1]
            V_ri = self.running_cov[:, 2]

        # --- 2. Whiten the Data ---
        # Add epsilon for numerical stability
        V_rr = V_rr + self.eps
        V_ii = V_ii + self.eps

        # Calculate the inverse square root of the covariance matrix
        # (V_rr, V_ri)
        # (V_ri, V_ii)
        det = V_rr * V_ii - V_ri**2
        s = torch.sqrt(det)
        t = torch.sqrt(V_rr + V_ii + 2 * s)
        
        # Prevent division by zero
        inv_t = 1.0 / (t + self.eps)
        
        Rrr = (V_ii + s) * inv_t
        Rii = (V_rr + s) * inv_t
        Rri = -V_ri * inv_t

        # Reshape for broadcasting
        Rrr = Rrr.view(1, self.num_features, 1, 1)
        Rii = Rii.view(1, self.num_features, 1, 1)
        Rri = Rri.view(1, self.num_features, 1, 1)

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

        # Apply the learnable transformation
        out_real = gamma_rr * whitened_x.real + gamma_ri * whitened_x.imag + beta_r
        out_imag = gamma_ri * whitened_x.real + gamma_ii * whitened_x.imag + beta_i

        return torch.complex(out_real, out_imag)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_features}, eps={self.eps}, momentum={self.momentum}, track_running_stats={self.track_running_stats})'
