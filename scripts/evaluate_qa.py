#!/usr/bin/env python
"""In-domain FrameNet question answering on the test split of the released dataset.

Uses the training prompt format (Alpaca style) and greedy decoding, as in the
original Evaluations/QA scripts:

    python scripts/evaluate_qa.py --task closed --adapted     # FrameLLaMA
    python scripts/evaluate_qa.py --task closed --baseline    # base model
    python scripts/evaluate_qa.py --task closed               # both

Predictions are written to results/runs/qa/<model>_<task>.csv
(e.g. baseline_closed.csv, as the released file).

The test split (247.9k held-out instances) has one file per question type,
read directly from Hugging Face (evaluation.qa.test_files in the config):

    --task opened   test/Opened.json   59.4k open-ended    (256 new tokens)
    --task closed   test/Closed.json   81.7k yes/no        (32 new tokens)
    --task mcq      test/MCQ.json      106.8k multiple choice (256 new tokens)

Any JSON/JSONL file with Question, Answer and optionally Instruction can be
given with --test-file instead.
"""

import argparse
import csv
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama.config import (add_model_args, cli_path, data_source, load_config, load_json_data,  # noqa: E402
                               repo_path, section, select_models)

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

MAX_TOKENS = {"closed": 32, "opened": 256, "mcq": 256}


def extract_final_answer(text, prompt="", valid_answers=["Yes", "No"]):
    """Extracts the Yes/No answer from model output, or returns the cleaned text."""
    cleaned_text = text.replace(prompt, "").strip()
    tokens = re.split(r'\s|\n', cleaned_text)
    for tok in reversed(tokens):
        tok_clean = tok.strip().capitalize().strip(string.punctuation)
        if tok_clean in valid_answers:
            return tok_clean
    return cleaned_text


def score_files(paths):
    """Yes/No accuracy of existing prediction CSVs."""
    from collections import Counter
    for path in paths:
        path = cli_path(path)
        if not path.exists():
            sys.exit(f"ERROR: prediction file not found: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or not {"Prediction", "References"} <= set(rows[0]):
            sys.exit(f"ERROR: {path} needs the columns Prediction and References.")
        correct = sum(r["Prediction"].strip() == r["References"].strip() for r in rows)
        print(f"{path.name}: accuracy {correct / len(rows):.4f} (n={len(rows):,})")
        print("  references:", dict(Counter(r["References"].strip() for r in rows)))
        print("  predictions:", dict(Counter(r["Prediction"].strip() for r in rows).most_common(4)))


def main():
    p = argparse.ArgumentParser(description="FrameNet QA evaluation.")
    p.add_argument("--config", default="configs/acl2026.yaml")
    p.add_argument("--task", choices=list(MAX_TOKENS), default=None)
    p.add_argument("--score-file", nargs="+", default=None,
                   help="Score existing yes/no prediction CSVs (columns Prediction, References), e.g. "
                        "results/acl2026/qa/baseline_closed.csv (no GPU needed)")
    p.add_argument("--test-file", default=None, help="Override evaluation.qa.test_files.<task>")
    add_model_args(p)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--base-model", default=None)
    args = p.parse_args()
    if args.score_file:
        return score_files(args.score_file)
    if not args.task:
        p.error("--task is required (closed, opened or mcq)")

    cfg = load_config(args.config)
    ev = section(cfg, "evaluation")
    qa_cfg = section(ev, "qa")
    spec = args.test_file or (qa_cfg.get("test_files") or {}).get(args.task)
    if spec is None:
        sys.exit(f"ERROR: no test file for task '{args.task}'. Pass --test-file.")
    test_file = data_source(spec, from_cli=bool(args.test_file), what="QA test file")
    out_dir = cli_path(args.output_dir) if args.output_dir else repo_path(ev.get("output_dir", "results/runs")) / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from vllm import SamplingParams
    except ImportError:
        sys.exit("ERROR: vLLM is not installed. Run: pip install -r requirements.txt "
                 "(vLLM needs Linux and an NVIDIA GPU).")
    from framellama.vllm_utils import generate, load_llm

    dataset = load_json_data(test_file)
    print(f"{len(dataset):,} '{args.task}' questions in {test_file}")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    prompts = [ALPACA_PROMPT.format(s.get("Instruction", "") or "", s.get("Question", "")) for s in dataset]
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS[args.task])
    vcfg = section(qa_cfg, "vllm") or {"gpu_memory_utilization": 0.85, "max_model_len": 8192, "dtype": "float16"}

    for key, model in select_models(args, ev):
        print(f"\n######## Model: {key} -> {model}")
        llm, lora = load_llm(model, base_model=args.base_model, disable_log_stats=True, **vcfg)
        outputs = generate(llm, prompts, params, lora)
        out = out_dir / f"{key}_{args.task}.csv"   # e.g. baseline_closed.csv, as released
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if args.task == "closed":
                w.writerow(["Question", "Prediction_raw", "Prediction", "References"])
            else:
                w.writerow(["Question", "Predictions", "References"])
            correct = 0
            for s, prompt, o in zip(dataset, prompts, outputs):
                raw = o.outputs[0].text.strip()
                ref = (s.get("Answer") or "").strip()
                if args.task == "closed":
                    pred = extract_final_answer(raw, prompt)
                    correct += pred == ref
                    w.writerow([s["Question"], raw, pred, ref])
                else:
                    w.writerow([s["Question"], raw, ref])
        if args.task == "closed":
            print(f"{key}: Yes/No accuracy {correct / len(dataset):.4f} (n={len(dataset)})")
        print(f"Predictions written to {out}")
        del llm


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
