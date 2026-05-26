import wandb
import torch
import os
import time
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.adamw import AdamW
from cs336_basics.transformer import Transformer
from cs336_basics.data_loading import data_loading
from cs336_basics.config import Config
import numpy as np

dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def evaluate(model, valid_dataset, config: Config):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config.eval_iters):
            x, y = data_loading(
                valid_dataset,
                config.batch_size,
                config.context_length,
                config.device,
            )

            logits = model(x)
            loss = cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
            )
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def train(config: Config):
    # wandb 设置
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.model_dump(),
        mode=config.wandb_mode,
        dir=config.wandb_dir,
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("valid/*", step_metric="train/step")
    run.define_metric("time/*", step_metric="train/step")

    try:
        # 加载训练数据集
        train_dataset = np.load(config.train_data_path, mmap_mode="r")
        valid_dataset = np.load(config.valid_data_path, mmap_mode="r")

        # 初始化各组件
        model = Transformer(
            vocab_size=config.vocab_size, 
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=torch.device(config.device),
            dtype=dtype_map[config.dtype]
        )

        optimizer = AdamW(
            params=model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
        start_step = 0
        # 恢复checkpoint
        if config.resume_path is not None:
            last_step = load_checkpoint(config.resume_path, model, optimizer)
            start_step = last_step + 1
        
        train_start_time = time.perf_counter()
        last_log_time = train_start_time
        best_valid_loss = float("inf")
        tokens_per_step = config.batch_size * config.context_length

        step = start_step - 1
        for step in range(start_step, config.num_steps):
            # 更新学习率
            lr = get_lr_cosine_schedule(
                t=step,
                max_learning_rate=config.max_lr,
                min_learning_rate=config.min_lr,
                warmup_iters=config.warmup_iters,
                cosine_cycle_iters=config.cosine_cycle_iters
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            # 将模型切换到训练模式
            model.train()
            # 随机采样训练数据
            x, y = data_loading(train_dataset, config.batch_size, config.context_length, config.device)
            # 前向传播
            logits = model(x)
            # 计算损失值
            loss = cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), config.max_l2_norm)  # ty:ignore[unresolved-attribute]
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
                valid_loss = evaluate(model, valid_dataset, config)
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
                run.log(
                    {
                        "checkpoint/step": step,
                    },
                    step=step,
                )
        run.summary["final_step"] = step
        run.summary["best_valid_loss"] = best_valid_loss
        run.summary["total_elapsed_sec"] = time.perf_counter() - train_start_time
        run.summary["total_tokens_seen"] = (step + 1) * tokens_per_step
    finally: 
        run.finish()

if __name__ == "__main__":
    config = Config()
    train(config)