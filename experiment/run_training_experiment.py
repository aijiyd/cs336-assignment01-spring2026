from __future__ import annotations

import argparse
import sys
from typing import Any

from cs336_basics.config import Config
from experiment.ablation_train import train_ablation
from experiment.experiment_specs import TRAIN_EXPERIMENT_BY_NAME, TRAIN_EXPERIMENTS


MODEL_SWITCH_KEYS = {"use_rmsnorm", "norm_position", "use_rope", "ffn_type"}


def _supported_config_fields() -> set[str]:
    return set(Config.model_fields)


def _split_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    supported = _supported_config_fields()
    config_overrides = {key: value for key, value in overrides.items() if key in supported}
    unsupported = {key: value for key, value in overrides.items() if key not in supported}
    return config_overrides, unsupported


def build_config_and_model_switches(
    experiment_name: str,
    cli_overrides: dict[str, Any],
) -> tuple[Config, dict[str, Any]]:
    experiment = TRAIN_EXPERIMENT_BY_NAME[experiment_name]
    raw_config_overrides, unsupported = _split_overrides(experiment.overrides)

    if experiment.requires_model_switches:
        model_switches = {
            key: value for key, value in experiment.overrides.items() if key in MODEL_SWITCH_KEYS
        }
        config_overrides = {
            key: value for key, value in raw_config_overrides.items() if key not in MODEL_SWITCH_KEYS
        }
        unsupported = {
            key: value for key, value in unsupported.items() if key not in MODEL_SWITCH_KEYS
        }
    else:
        model_switches = {}
        config_overrides = raw_config_overrides

    if unsupported:
        missing = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Experiment {experiment.name!r} has unsupported config fields: {missing}."
        )

    return Config(**config_overrides, **cli_overrides), model_switches


def list_experiments() -> None:
    for experiment in TRAIN_EXPERIMENTS:
        print(f"{experiment.name:12s} [{experiment.group}] {experiment.description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a predefined CS336 training experiment.")
    parser.add_argument("--name", choices=sorted(TRAIN_EXPERIMENT_BY_NAME), help="Experiment name.")
    parser.add_argument("--list", action="store_true", help="List available experiments.")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Override runtime device.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="Override runtime dtype.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="Override W&B mode.")
    parser.add_argument("--resume-path", help="Resume from a checkpoint path.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without training.")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.name is None:
        parser.error("--name is required unless --list is passed")

    cli_overrides: dict[str, Any] = {}
    if args.device is not None:
        cli_overrides["device"] = args.device
    if args.dtype is not None:
        cli_overrides["dtype"] = args.dtype
    if args.wandb_mode is not None:
        cli_overrides["wandb_mode"] = args.wandb_mode
    if args.resume_path is not None:
        cli_overrides["resume_path"] = args.resume_path

    try:
        config, model_switches = build_config_and_model_switches(args.name, cli_overrides)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None

    if args.dry_run:
        print({"config": config.model_dump(), "model_switches": model_switches})
        return

    if model_switches:
        train_ablation(config, **model_switches)
    else:
        from cs336_basics.train import train

        train(config)


if __name__ == "__main__":
    main()
