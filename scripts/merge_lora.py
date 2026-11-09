#!/usr/bin/env python
"""Merge trained LoRA adapters into the base model (16-bit) for vLLM.

The evaluation scripts load a merged FP16 model with vLLM. This script
produces it from the adapters saved by scripts/train.py:

    python scripts/merge_lora.py --config configs/acl2026.yaml

or with explicit paths:

    python scripts/merge_lora.py --adapter-dir checkpoints/acl2026/final_adapter \\
                                 --output-dir checkpoints/acl2026/merged_fp16

The released checkpoint on Hugging Face is already usable directly; this step
is only needed after re-training.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama.config import cli_path, load_config, repo_path, require_file, section  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapters into the base model.")
    p.add_argument("--config", default="configs/acl2026.yaml")
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = section(load_config(args.config), "training")
    adapter_dir = cli_path(args.adapter_dir) if args.adapter_dir else repo_path(cfg["adapter_dir"])
    output_dir = cli_path(args.output_dir) if args.output_dir else repo_path(cfg["merged_dir"])
    require_file(adapter_dir / "adapter_config.json", "LoRA adapter",
                 "Run scripts/train.py first or pass --adapter-dir.")

    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError as exc:
        sys.exit(f"ERROR: missing dependency ({exc}). Install requirements-train.txt.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        dtype=torch.float16,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    print(f"Merged FP16 model saved to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
