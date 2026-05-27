from __future__ import annotations

import argparse
import os

import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.config import Config
from cs336_basics.decoder import decode
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import Transformer
from cs336_basics.train import dtype_map
from experiment.experiment_specs import GENERATION_PROMPTS, GENERATION_SETTINGS


def build_model(config: Config) -> Transformer:
    return Transformer(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=torch.device(config.device),
        dtype=dtype_map[config.dtype],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text samples from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", default="data/generations", help="Directory for generated text files.")
    parser.add_argument("--vocab-path", default="data/tinystories_vocab.pkl")
    parser.add_argument("--merges-path", default="data/tinystories_merges.pkl")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    config_overrides = {}
    if args.device is not None:
        config_overrides["device"] = args.device
    if args.dtype is not None:
        config_overrides["dtype"] = args.dtype
    config = Config(**config_overrides)

    tokenizer = BPETokenizer.from_file(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps)
    iteration = load_checkpoint(args.checkpoint, model, optimizer)

    os.makedirs(args.output_dir, exist_ok=True)
    for setting in GENERATION_SETTINGS:
        output_path = os.path.join(args.output_dir, f"{setting['name']}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# checkpoint: {args.checkpoint}\n")
            f.write(f"# iteration: {iteration}\n")
            f.write(f"# temperature: {setting['temperature']}\n")
            f.write(f"# top_p: {setting['top_p']}\n\n")
            for prompt in GENERATION_PROMPTS:
                text = decode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    context_length=config.context_length,
                    device=config.device,
                    temperature=setting["temperature"],
                    top_p=setting["top_p"],
                )
                f.write(f"## Prompt\n{prompt}\n\n")
                f.write(f"## Output\n{text}\n\n")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
