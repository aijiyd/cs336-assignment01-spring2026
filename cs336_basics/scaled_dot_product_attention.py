from cs336_basics.softmax import softmax
import math
from einops import einsum
import torch
from torch import nn

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None=None):
        d_k = Q.shape[-1]
        scores = einsum(Q, K, "batch_size ... queries d_k, batch_size ... keys d_k -> batch_size ... queries keys") / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        return einsum(softmax(scores, dim=-1), V, "batch_size ... queries keys, batch_size ... keys d_v -> batch_size ... queries d_v")


