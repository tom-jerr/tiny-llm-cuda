import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.to(dtype=torch.float32)
        weight_float = self.weight.to(dtype=torch.float32)
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x_float * rms
        return (x_normed * weight_float).to(dtype=original_dtype)
