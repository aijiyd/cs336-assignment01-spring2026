import torch
import random
import numpy.typing as npt
import numpy as np

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]

    x_np = np.stack([dataset[s: s + context_length] for s in starts])
    y_np = np.array([dataset[s + 1: s + context_length + 1] for s in starts])
    x = torch.from_numpy(x_np).long().to(device)
    y = torch.from_numpy(y_np).long().to(device)
    return x, y