from einops import einsum
import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        device: torch.device | None=None, 
        dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty(self.out_dim, self.in_dim, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_dim + out_dim))
        nn.init.trunc_normal_(self.weight, 0.0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, d_in] @ [d_out, d_in] -> [batch_size, seq_len, d_out]
        return einsum(x, self.weight, "batch_size seq_len d_in, d_out d_in -> batch_size seq_len d_out")

