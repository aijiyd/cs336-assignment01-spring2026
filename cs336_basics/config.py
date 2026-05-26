import torch
from pydantic import BaseModel, ConfigDict
from typing import Literal


class Config(BaseModel):
    model_config = ConfigDict()

    # data
    train_data_path: str = "data/tinystories_train_tokens.npy"
    valid_data_path: str = "data/tinystories_valid_tokens.npy"

    # output
    checkpoint_dir: str = "checkpoints"
    latest_checkpoint_path: str = "checkpoints/latest.pt"
    resume_path: str | None = None

    # model
    vocab_size: int = 10000
    context_length: int = 512 # cpu:256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344 # 8/3 d_model
    rope_theta: float = 10000.0

    # training
    # batch size × total step count × context length = 327,680,000
    batch_size: int = 64 # cpu:32
    num_steps: int = 10000 # cpu:5000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 1000
    cosine_cycle_iters: int = 10000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_l2_norm: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 20

    # save
    save_interval: int = 1000

    # wandb
    wandb_project: str = "cs336-assignment1"
    wandb_run_name: str = "tinystories-transformer-512d-4l"
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_dir: str = "wandb"

    # runtime
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"