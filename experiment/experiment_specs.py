from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ExperimentGroup = Literal["baseline", "architecture", "batch_size"]


@dataclass(frozen=True)
class TrainExperiment:
    name: str
    group: ExperimentGroup
    description: str
    overrides: dict[str, Any]
    requires_model_switches: bool = False


BASE_TOTAL_TOKENS = 64 * 512 * 10_000


BASELINE = TrainExperiment(
    name="baseline",
    group="baseline",
    description="RMSNorm + pre-norm + RoPE + SwiGLU.",
    overrides={
        "batch_size": 64,
        "num_steps": 10_000,
        "context_length": 512,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "wandb_run_name": "baseline",
        "checkpoint_dir": "checkpoints/baseline",
        "latest_checkpoint_path": "checkpoints/baseline/latest.pt",
    },
)


ARCHITECTURE_EXPERIMENTS = [
    TrainExperiment(
        name="no_rmsnorm",
        group="architecture",
        description="Remove RMSNorm.",
        requires_model_switches=True,
        overrides={
            "use_rmsnorm": False, # 移除 RMSNorm
            "norm_position": "pre",
            "use_rope": True,
            "ffn_type": "swiglu",
            "wandb_run_name": "no_rmsnorm",
            "checkpoint_dir": "checkpoints/no_rmsnorm",
            "latest_checkpoint_path": "checkpoints/no_rmsnorm/latest.pt",
        },
    ),
    TrainExperiment(
        name="post_norm",
        group="architecture",
        description="Use post-norm Transformer blocks.",
        requires_model_switches=True,
        overrides={
            "use_rmsnorm": True,
            "norm_position": "post", # 后归一化
            "use_rope": True,
            "ffn_type": "swiglu",
            "wandb_run_name": "post_norm",
            "checkpoint_dir": "checkpoints/post_norm",
            "latest_checkpoint_path": "checkpoints/post_norm/latest.pt",
        },
    ),
    TrainExperiment(
        name="nope",
        group="architecture",
        description="Replace RoPE with no positional encoding.",
        requires_model_switches=True,
        overrides={
            "use_rmsnorm": True,
            "norm_position": "pre",
            "use_rope": False, # 移除位置编码
            "ffn_type": "swiglu",
            "wandb_run_name": "nope",
            "checkpoint_dir": "checkpoints/nope",
            "latest_checkpoint_path": "checkpoints/nope/latest.pt",
        },
    ),
    TrainExperiment(
        name="silu_ffn",
        group="architecture",
        description="Replace SwiGLU FFN with SiLU FFN and use d_ff=4*d_model.",
        requires_model_switches=True,
        overrides={
            "use_rmsnorm": True,
            "norm_position": "pre",
            "use_rope": True,
            "ffn_type": "silu", # ffn改成silu
            "d_ff": 2048,
            "wandb_run_name": "silu_ffn",
            "checkpoint_dir": "checkpoints/silu_ffn",
            "latest_checkpoint_path": "checkpoints/silu_ffn/latest.pt",
        },
    ),
]


def _steps_for_batch_size(batch_size: int) -> int:
    return BASE_TOTAL_TOKENS // (batch_size * 512)


BATCH_SIZE_EXPERIMENTS = [
    TrainExperiment(
        name=f"bs_{batch_size}",
        group="batch_size",
        description=f"Baseline architecture with batch_size={batch_size}.",
        overrides={
            "batch_size": batch_size,
            "num_steps": _steps_for_batch_size(batch_size),
            "context_length": 512,
            "wandb_run_name": f"bs_{batch_size}",
            "checkpoint_dir": f"checkpoints/bs_{batch_size}",
            "latest_checkpoint_path": f"checkpoints/bs_{batch_size}/latest.pt",
        },
    )
    for batch_size in (16, 32, 64, 128)
]


TRAIN_EXPERIMENTS = [
    BASELINE,
    *ARCHITECTURE_EXPERIMENTS,
    *BATCH_SIZE_EXPERIMENTS,
]


TRAIN_EXPERIMENT_BY_NAME = {experiment.name: experiment for experiment in TRAIN_EXPERIMENTS}


GENERATION_PROMPTS = [
    "Once upon a time",
    "One day, Lily found",
    "Tom was very happy because",
    "The little dog wanted to",
]


GENERATION_SETTINGS = [
    {"name": "greedy", "temperature": 0.0, "top_p": None},
    {"name": "temp_07_top_p_09", "temperature": 0.7, "top_p": 0.9},
    {"name": "temp_10_top_p_09", "temperature": 1.0, "top_p": 0.9},
    {"name": "temp_12_top_p_095", "temperature": 1.2, "top_p": 0.95},
]
