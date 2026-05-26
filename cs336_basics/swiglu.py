from cs336_basics.linear import Linear
from einops import einsum
import math
import torch
from torch import nn

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self,  d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        # self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        # self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        # std = math.sqrt(2.0 / (d_model + d_ff))
        # nn.init.trunc_normal_(self.w1, 0.0, std, -3 * std, 3 * std)
        # nn.init.trunc_normal_(self.w2, 0.0, std, -3 * std, 3 * std)
        # nn.init.trunc_normal_(self.w3, 0.0, std, -3 * std, 3 * std)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: torch.Tensor):
        # x: Float[Tensor, "... d_model"]
        # y1 = silu(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"))
        # y2 = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        # return einsum(self.w2, y1 * y2, "d_model d_ff, ... d_ff -> ... d_model")
        return self.w2(silu(self.w1(x)) * self.w3(x))


