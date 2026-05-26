import math
from einops import einsum
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        y = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        result = x / y * self.weight
        return result.to(in_dtype)