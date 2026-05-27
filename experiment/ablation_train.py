from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
import torch
import wandb
from torch import nn

from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.config import Config
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import data_loading
from cs336_basics.embedding import Embedding
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.linear import Linear
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU, silu


dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


NormPosition = Literal["pre", "post"]
FFNType = Literal["swiglu", "silu"]


class SiLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class AblationTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
        use_rmsnorm: bool,
        norm_position: NormPosition,
        use_rope: bool,
        ffn_type: FFNType,
    ):
        super().__init__()
        self.norm_position = norm_position
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=use_rope,
            device=device,
            dtype=dtype,
        )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_position == "pre":
            out = x + self.attn(self.norm1(x))
            return out + self.ffn(self.norm2(out))

        if self.norm_position == "post":
            out = self.norm1(x + self.attn(x))
            return self.norm2(out + self.ffn(out))

        raise ValueError(f"Unknown norm_position: {self.norm_position}")


class AblationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype,
        use_rmsnorm: bool,
        norm_position: NormPosition,
        use_rope: bool,
        ffn_type: FFNType,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                AblationTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.out = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        return self.out(out)


def evaluate_ablation(model: nn.Module, valid_dataset: np.ndarray, config: Config) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config.eval_iters):
            x, y = data_loading(valid_dataset, config.batch_size, config.context_length, config.device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def train_ablation(
    config: Config,
    *,
    use_rmsnorm: bool,
    norm_position: NormPosition,
    use_rope: bool,
    ffn_type: FFNType,
) -> None:
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            **config.model_dump(),
            "use_rmsnorm": use_rmsnorm,
            "norm_position": norm_position,
            "use_rope": use_rope,
            "ffn_type": ffn_type,
        },
        mode=config.wandb_mode,
        dir=config.wandb_dir,
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("valid/*", step_metric="train/step")
    run.define_metric("time/*", step_metric="train/step")

    try:
        train_dataset = np.load(config.train_data_path, mmap_mode="r")
        valid_dataset = np.load(config.valid_data_path, mmap_mode="r")

        model = AblationTransformer(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=torch.device(config.device),
            dtype=dtype_map[config.dtype],
            use_rmsnorm=use_rmsnorm,
            norm_position=norm_position,
            use_rope=use_rope,
            ffn_type=ffn_type,
        )
        optimizer = AdamW(
            params=model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        start_step = 0
        if config.resume_path is not None:
            last_step = load_checkpoint(config.resume_path, model, optimizer)
            start_step = last_step + 1

        train_start_time = time.perf_counter()
        last_log_time = train_start_time
        best_valid_loss = float("inf")
        tokens_per_step = config.batch_size * config.context_length

        step = start_step - 1
        for step in range(start_step, config.num_steps):
            lr = get_lr_cosine_schedule(
                t=step,
                max_learning_rate=config.max_lr,
                min_learning_rate=config.min_lr,
                warmup_iters=config.warmup_iters,
                cosine_cycle_iters=config.cosine_cycle_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            x, y = data_loading(train_dataset, config.batch_size, config.context_length, config.device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), config.max_l2_norm)
            optimizer.step()

            now = time.perf_counter()
            step_time_sec = now - last_log_time
            elapsed_sec = now - train_start_time
            last_log_time = now
            tokens_seen = (step + 1) * tokens_per_step
            tokens_per_sec = tokens_per_step / step_time_sec

            log_dict = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/step": step,
                "train/tokens_seen": tokens_seen,
                "time/step_time_sec": step_time_sec,
                "time/elapsed_sec": elapsed_sec,
                "time/tokens_per_sec": tokens_per_sec,
            }

            if step % config.eval_interval == 0:
                valid_loss = evaluate_ablation(model, valid_dataset, config)
                best_valid_loss = min(best_valid_loss, valid_loss)
                log_dict["valid/loss"] = valid_loss
                log_dict["valid/best_loss"] = best_valid_loss

            run.log(log_dict, step=step)
            if step % 10 == 0:
                msg = (
                    f"step={step}/{config.num_steps} "
                    f"train_loss={loss.item():.4f} "
                    f"lr={lr:.2e} "
                    f"tokens_per_sec={tokens_per_sec:.0f} "
                    f"elapsed={elapsed_sec / 60:.1f}min"
                )
                if "valid/loss" in log_dict:
                    msg += f" valid_loss={log_dict['valid/loss']:.4f}"
                print(msg, flush=True)

            if step % config.save_interval == 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                save_checkpoint(model, optimizer, step, config.latest_checkpoint_path)
                run.log({"checkpoint/step": step}, step=step)

        # 训练结束后强制保存最终模型，避免最后一步不是 save_interval 的整数倍时只保存到旧 checkpoint。
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        save_checkpoint(model, optimizer, step, config.latest_checkpoint_path)
        run.log({"checkpoint/step": step}, step=step)
        print(f"saved final checkpoint at step={step}", flush=True)

        run.summary["final_step"] = step
        run.summary["best_valid_loss"] = best_valid_loss
        run.summary["total_elapsed_sec"] = time.perf_counter() - train_start_time
        run.summary["total_tokens_seen"] = (step + 1) * tokens_per_step
    finally:
        run.finish()
