from einops import einsum
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, 0.0, 1.0, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[batch, seq_len], [vocab_size, d_model] -> [batch, seq_len, d_model]"""
        return self.weight[token_ids]
