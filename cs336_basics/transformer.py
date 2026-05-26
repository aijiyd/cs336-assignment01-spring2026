from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.swiglu import SwiGLU
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from torch import nn
import torch

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.atten = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, True, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor):
        out = self.atten(self.norm1(x)) + x
        return out + self.ffn(self.norm2(out))


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length:int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.out = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x: torch.Tensor):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        out = self.out(out)
        return out