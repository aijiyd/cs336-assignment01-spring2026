# CS336 Experiment Scripts

这些脚本只负责实验组织和结果导出，不替代作业里的模型实现。

## 1. 查看实验列表

```bash
uv run python -m experiment.run_training_experiment --list
```

## 2. 运行 baseline

```bash
mkdir -p logs checkpoints wandb
uv run python -u -m experiment.run_training_experiment --name baseline 2>&1 | tee logs/baseline.log
```

AutoDL 网络不稳时可以加 `--wandb-mode offline`：

```bash
uv run --no-sync python -u -m experiment.run_training_experiment --name baseline --wandb-mode offline 2>&1 | tee logs/baseline.log
```

## 3. 运行 batch size 实验

这些实验保持总 token 数一致。

```bash
uv run python -u -m experiment.run_training_experiment --name bs_16 2>&1 | tee logs/bs_16.log
uv run python -u -m experiment.run_training_experiment --name bs_32 2>&1 | tee logs/bs_32.log
uv run python -u -m experiment.run_training_experiment --name bs_64 2>&1 | tee logs/bs_64.log
uv run python -u -m experiment.run_training_experiment --name bs_128 2>&1 | tee logs/bs_128.log
```

如果 `bs_128` 显存不够，跳过即可。

## 4. 结构消融实验

已定义：

- `no_rmsnorm`
- `post_norm`
- `nope`
- `silu_ffn`

这些实验使用 `experiment.ablation_train` 里的实验专用模型，不修改 `cs336_basics/transformer.py`。

运行方式：

```bash
uv run python -u -m experiment.run_training_experiment --name no_rmsnorm
uv run python -u -m experiment.run_training_experiment --name post_norm
uv run python -u -m experiment.run_training_experiment --name nope
uv run python -u -m experiment.run_training_experiment --name silu_ffn
```

## 5. 生成文本

用训练好的 checkpoint 生成多组 temperature/top-p 结果：

```bash
uv run python -u -m experiment.generate_samples --checkpoint checkpoints/baseline/latest.pt
```

输出目录：

```text
data/generations/
```

## 6. 学习率曲线

```bash
uv run python -u -m experiment.plot_lr_curve
```

输出：

```text
data/figures/lr_curve.csv
data/figures/lr_curve.png
```

如果环境里没有 `matplotlib`，脚本仍会写出 CSV，只会跳过 PNG。
