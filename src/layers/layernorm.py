import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", weight)

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # if mul_ here, it will cause precision issues in float32
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_forward(x)
