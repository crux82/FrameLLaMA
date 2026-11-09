#!/usr/bin/env python
"""NLI evaluation on CONFER and the SNLI diagnostic subset (zero- and few-shot).

Models: --adapted (FrameLLaMA) and --baseline (Llama-3.1-8B-Instruct);
both when neither is given.

    python scripts/evaluate_nli.py --adapted                 # FrameLLaMA, all benchmarks/settings
    python scripts/evaluate_nli.py --baseline                # base model
    python scripts/evaluate_nli.py --adapted --baseline      # both (same as no flag)

Common variations:

    # FrameLLaMA, CONFER, zero-shot
    python scripts/evaluate_nli.py --adapted --benchmarks confer --settings zero-shot

    # any other model (Hugging Face id or local folder)
    python scripts/evaluate_nli.py --models path/to/model

    # gpt-oss baselines use their own prompt format
    python scripts/evaluate_nli.py --config configs/acl2026.yaml \\
        --models openai/gpt-oss-20b --prompt-format gpt-oss

    # check data and prompts without a GPU
    python scripts/evaluate_nli.py --config configs/acl2026.yaml --dry-run

    # score the released prediction files (no GPU)
    python scripts/evaluate_nli.py --score-file results/acl2026/nli/*.csv

For every run the script writes <model>_nli_<0shot|fewshot>_<benchmark>.csv
(e.g. adapted_nli_0shot_confer.csv; columns Premise, Hypothesis, Predictions,
References, Raw) and a .metrics.json file into results/runs/nli/, and prints
accuracy, weighted P/R/F1 and per-class scores.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama import nli  # noqa: E402
from framellama.config import add_model_args, cli_path, load_config, repo_path, require_file, section, select_models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="NLI evaluation (ACL 2026 protocol).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", default="configs/acl2026.yaml")
    p.add_argument("--benchmarks", nargs="+", choices=["confer", "snli"], default=None)
    p.add_argument("--settings", nargs="+", choices=["zero-shot", "few-shot"], default=None)
    p.add_argument("--prompt-format", choices=["raw", "gpt-oss"], default="raw")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="Use only the first N examples (smoke test)")
    p.add_argument("--base-model", default=None, help="Base model if a checkpoint is a LoRA adapter")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Build prompts and stop (no GPU needed)")
    p.add_argument("--score-file", nargs="+", default=None,
                   help="Score existing prediction CSVs (columns Predictions, References), e.g. "
                        "results/acl2026/nli/adapted_nli_0shot_confer.csv (no GPU needed)")
    add_model_args(p)
    return p.parse_args()



def print_metrics(name, m):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {m['accuracy']:.4f}   (n={m['n']})")
    print(f"F1 (w)   : {m['f1_weighted']:.4f}")
    print(f"Recall(w): {m['recall_weighted']:.4f}")
    print(f"Prec. (w): {m['precision_weighted']:.4f}")
    for label, s in m["per_class"].items():
        if isinstance(s, dict) and label not in ("macro avg", "weighted avg"):
            print(f"  {label:14s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1-score']:.3f} n={int(s['support'])}")
    print("Predicted label distribution:", m["prediction_distribution"])


def score_files(paths):
    """Accuracy, weighted P/R/F1 and per-class scores of existing prediction CSVs."""
    for path in paths:
        path = cli_path(path)
        require_file(path, "Prediction file")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or not {"Predictions", "References"} <= set(rows[0]):
            sys.exit(f"ERROR: {path} needs the columns Predictions and References.")
        preds = [r["Predictions"].strip() for r in rows]
        refs = [r["References"].strip() for r in rows]
        print_metrics(path.name, nli.metrics(preds, refs))


def main():
    args = parse_args()
    if args.score_file:
        return score_files(args.score_file)
    cfg = load_config(args.config)
    ev = section(cfg, "evaluation")
    nli_cfg = section(ev, "nli")
    vcfg = {**section(ev, "vllm"), **section(nli_cfg, "vllm")}

    models = select_models(args, ev)
    benchmarks = args.benchmarks or list(nli_cfg.get("benchmarks", {}))
    settings = args.settings or nli_cfg.get("settings", ["zero-shot", "few-shot"])
    out_dir = cli_path(args.output_dir) if args.output_dir else repo_path(ev.get("output_dir", "results/runs")) / "nli"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for b in benchmarks:
        path = repo_path(nli_cfg["benchmarks"][b])
        require_file(path, f"{b} test file", "See evaluation/README.md for how to obtain it.")
        data[b] = nli.load_nli_file(path)[: args.limit] if args.limit else nli.load_nli_file(path)
        print(f"Loaded {b}: {len(data[b])} examples from {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

    if args.dry_run:
        for b in benchmarks:
            for s in settings:
                prompts = nli.build_prompts(data[b], s, b, args.prompt_format)
                print(f"\n--- {b} / {s} / {args.prompt_format}: {len(prompts)} prompts; first one ---")
                print(prompts[0])
                print("--- decoding:", nli.sampling_kwargs(s, args.prompt_format))
        print("\nDry run finished (no model loaded).")
        return

    try:
        from vllm import SamplingParams
    except ImportError:
        sys.exit("ERROR: vLLM is not installed. Run: pip install -r requirements.txt "
                 "(vLLM needs Linux and an NVIDIA GPU).")
    from framellama.vllm_utils import generate, load_llm

    summary = []
    for key, model in models:
        print(f"\n######## Model: {key} -> {model}")
        llm, lora = load_llm(
            model,
            gpu_memory_utilization=args.gpu_memory_utilization or vcfg.get("gpu_memory_utilization", 0.80),
            max_model_len=args.max_model_len or vcfg.get("max_model_len", 8192),
            dtype=vcfg.get("dtype", "auto"),
            base_model=args.base_model,
        )
        for b in benchmarks:
            for s in settings:
                items = data[b]
                prompts = nli.build_prompts(items, s, b, args.prompt_format)
                params = SamplingParams(**nli.sampling_kwargs(s, args.prompt_format))
                outputs = generate(llm, prompts, params, lora)
                raw = [o.outputs[0].text for o in outputs]
                preds, refs = nli.score_outputs(raw, items, s, args.prompt_format)
                m = nli.metrics(preds, refs)
                setting_tag = "0shot" if s == "zero-shot" else "fewshot"
                run = f"{key}_nli_{setting_tag}_{b}" + ("_gptoss" if args.prompt_format == "gpt-oss" else "")
                with open(out_dir / f"{run}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["Premise", "Hypothesis", "Predictions", "References", "Raw"])
                    for it, p, r, t in zip(items, preds, refs, raw):
                        w.writerow([it["premise"], it["hypothesis"], p, r, t])
                m.update(model=model, benchmark=b, setting=s, prompt_format=args.prompt_format)
                with open(out_dir / f"{run}.metrics.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2)
                print_metrics(run, m)
                summary.append((key, b, s, m["accuracy"], m["f1_weighted"]))
        del llm

    print("\n======== Summary ========")
    print(f"{'model':28s} {'benchmark':8s} {'setting':10s} {'acc':>7s} {'F1(w)':>7s}")
    for k, b, s, acc, f1 in summary:
        print(f"{k[:28]:28s} {b:8s} {s:10s} {acc:7.4f} {f1:7.4f}")
    print(f"\nPredictions and metrics written to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
