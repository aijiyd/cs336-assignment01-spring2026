import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: torch.device | None=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0

        inv_freq = 1.0 / (self.theta**(torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / self.d_k))
        positions = torch.arange(0, self.max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq) # (max_seq_len, d_k / 2)
        
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False) # 不写入模型参数中
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_odd = x[...,1::2] # 同一个token的奇数维度
        x_even = x[..., 0::2] # 同一个token的偶数维度

        cos = self.cos_cached[token_positions]  # ty:ignore[not-subscriptable]
        sin = self.sin_cached[token_positions]  # ty:ignore[not-subscriptable]

        out_odd = sin * x_even + cos * x_odd
        out_even = cos * x_even - sin * x_odd

        out = torch.empty_like(x)
        out[..., 1::2] = out_odd
        out[..., 0::2] = out_even
        return out