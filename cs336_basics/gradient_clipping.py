import math
import torch
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = list(parameters)
    eps = 1e-6
    total_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total_norm += param.grad.pow(2).sum()
    total_norm = math.sqrt(total_norm)
    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad *= scale
