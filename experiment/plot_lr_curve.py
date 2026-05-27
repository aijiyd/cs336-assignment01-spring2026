from __future__ import annotations

import argparse
import csv
import os

from cs336_basics.config import Config
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the cosine learning-rate schedule.")
    parser.add_argument("--output", default="data/figures/lr_curve.png")
    parser.add_argument("--csv-output", default="data/figures/lr_curve.csv")
    args = parser.parse_args()

    config = Config()
    steps = list(range(config.num_steps))
    learning_rates = [
        get_lr_cosine_schedule(
            t=step,
            max_learning_rate=config.max_lr,
            min_learning_rate=config.min_lr,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters,
        )
        for step in steps
    ]

    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "learning_rate"])
        writer.writerows(zip(steps, learning_rates))
    print(f"wrote {args.csv_output}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; CSV was written, PNG was skipped.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, learning_rates)
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.title("Cosine learning-rate schedule")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
