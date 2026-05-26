from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear
import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                d_model: int, 
                num_heads: int, 
                max_seq_len: int | None=None,
                theta: float | None=None,
                use_rope: bool=False, 
                device: torch.device | None=None, 
                dtype: torch.dtype | None=None
            ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)  

        if self.use_rope and theta and max_seq_len:
            self.rope = RoPE(self.head_dim, theta, max_seq_len, device)
        else:
            self.rope = None
            
    def _split_heads(self, x :torch.Tensor):
        *batch_dim, seq_len, _ = x.size()
        # ...,seq_len,d_model->...,seq_len,num_heads,head_dim->...,num_heads,seq_len,head_dim
        outputs = x.reshape(*batch_dim, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        return outputs
    
    def _merge_heads(self, x: torch.Tensor):
        *batch_dim, _, seq_len, _ = x.shape
        # ...,num_heads,seq_len,head_dim->...,seq_len,num_heads,head_dim->...,seq_len,d_model
        return x.transpose(-2, -3).reshape(*batch_dim, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None):
        *batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                base_position = torch.arange(seq_len, device=x.device, dtype=torch.long) # [seq_len]
                token_positions = base_position.view(*([1] * len(batch)), seq_len).expand(*(batch), seq_len) # ... seq_len
            else:
                token_positions = token_positions.to(device=x.device, dtype=torch.long)
            
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        masks = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        output = scaled_dot_product_attention(q, k, v, masks)
        outputs = self.o_proj(self._merge_heads(output))
        return outputs