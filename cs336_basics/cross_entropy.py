from functorch.dim import index
from cs336_basics.softmax import softmax
import torch

def cross_entropy(input: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # input: batch_size vocab_size, targets: batch_size
    x = input - input.max(dim=-1, keepdim=True).values # 保证数值稳定性
    logits = torch.log(torch.sum(torch.exp(x), dim=-1))
    target_logits = x.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    out = -target_logits + logits
    return out.mean()




