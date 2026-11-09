#!/usr/bin/env python
"""SRL evaluation: frame elements (roles + spans), lexical units, frames.

Models: --adapted (FrameLLaMA) and --baseline (Llama-3.1-8B-Instruct);
both when neither is given.

    python scripts/evaluate_srl.py --adapted                 # FrameLLaMA, all tasks
    python scripts/evaluate_srl.py --baseline --tasks roles  # base model, frame elements

Tasks
  roles  given sentence, frame and lexical unit, predict frame elements and spans
  lu     given sentence and frame, predict the lexical unit
  frame  given sentence and lexical unit, predict the frame

For each model and task the script
  1. builds one query per gold annotation of evaluation/splits/srl/srl_gold_std.json,
  2. generates answers with vLLM (greedy, max 512 tokens) and parses the JSON,
  3. writes <model>_srl_result_<task>.json (model annotations per sentence)
     and <model>_srl_gold-std_<task>.json (the matching gold annotations),
     e.g. baseline_srl_result_roles.json, as in the released files
     (task tags: roles, LUs, frames),
  4. scores them with the original F1 code.

Scoring files that already exist (no GPU needed):

    python scripts/evaluate_srl.py --score-only --task roles \\
        --result-file results/acl2026/srl/baseline_srl_result_roles.json

Without --gold-file, each predicted annotation is matched to the gold
annotation of evaluation/splits/srl/srl_gold_std.json with the same
sentence, frame and lexical unit.

Check inputs and prompts without a GPU:

    python scripts/evaluate_srl.py --config configs/acl2026.yaml --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama import srl  # noqa: E402
from framellama.config import add_model_args, cli_path, load_config, repo_path, require_file, section, select_models  # noqa: E402

# task -> tag in the output file names (as in the released files: *_srl_result_roles.json, *_srl_result_LUs.json)
TASK_TAG = {"roles": "roles", "lu": "LUs", "frame": "frames"}


def parse_args():
    p = argparse.ArgumentParser(description="SRL evaluation (ACL 2026 protocol).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", default="configs/acl2026.yaml")
    p.add_argument("--tasks", nargs="+", choices=srl.TASKS, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="Use only the first N gold sentences")
    p.add_argument("--base-model", default=None, help="Base model if a checkpoint is a LoRA adapter")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Build queries and prompts, then stop")
    p.add_argument("--score-only", action="store_true", help="Score existing result/gold files")
    p.add_argument("--task", choices=["roles", "lu", "frame"], help="Task for --score-only")
    p.add_argument("--result-file", help="Model annotations (for --score-only)")
    p.add_argument("--gold-file", help="Aligned gold annotations (for --score-only). If omitted, the "
                                       "gold annotations are taken from srl.gold_file in the config")
    add_model_args(p)
    return p.parse_args()



def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def score_only(args):
    if not (args.task and args.result_file):
        sys.exit("ERROR: --score-only needs --task and --result-file (and optionally --gold-file)")
    res_path = cli_path(args.result_file)
    if args.gold_file:
        gold_path = cli_path(args.gold_file)
    else:
        cfg = section(section(load_config(args.config), "evaluation"), "srl")
        gold_path = repo_path(cfg["gold_file"])
    for p in (res_path, gold_path):
        require_file(p, "Input file")
        if p.stat().st_size and not p.read_bytes()[:4096].strip(b"\x00"):
            sys.exit(f"ERROR: {p} contains only zero bytes (the file is corrupted).")
    result, gold = load_json(res_path), load_json(gold_path)
    if not args.gold_file:
        result, gold = srl.align_to_gold(result, gold)
        print(f"Aligned with {gold_path.name}: {len(result)} sentences, "
              f"{sum(len(s['annotations']) for s in result)} annotations")
    m = srl.score(args.task, result, gold)
    print(json.dumps(m, indent=2))


def main():
    args = parse_args()
    if args.score_only:
        return score_only(args)

    cfg = load_config(args.config)
    ev = section(cfg, "evaluation")
    srl_cfg = section(ev, "srl")
    vcfg = {**section(ev, "vllm"), **section(srl_cfg, "vllm")}
    models = select_models(args, ev)
    tasks = args.tasks or srl_cfg.get("tasks", list(srl.TASKS))
    out_dir = cli_path(args.output_dir) if args.output_dir else repo_path(ev.get("output_dir", "results/runs")) / "srl"
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = repo_path(srl_cfg["gold_file"])
    require_file(gold_path, "SRL gold file")
    gold = srl.load_gold(gold_path)
    if args.limit:
        gold = gold[: args.limit]
    inputs = srl.prepare_inputs(gold)
    print(f"Gold: {len(gold)} sentences, {len(inputs)} annotations -> {len(inputs)} queries per task")

    if args.dry_run:
        for t in tasks:
            print(f"\n--- task {t}: first prompt ---")
            print(srl.build_prompt(t, inputs[0]))
        print("\n--- decoding:", srl.sampling_kwargs())
        print("Dry run finished (no model loaded).")
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
            gpu_memory_utilization=args.gpu_memory_utilization or vcfg.get("gpu_memory_utilization", 0.90),
            max_model_len=args.max_model_len or vcfg.get("max_model_len", 8192),
            dtype=vcfg.get("dtype", "float16"),
            base_model=args.base_model,
            disable_log_stats=True,
        )
        params = SamplingParams(**srl.sampling_kwargs())
        for t in tasks:
            tag = TASK_TAG[t]
            prompts = [srl.build_prompt(t, s) for s in inputs]
            outputs = generate(llm, prompts, params, lora)
            raw = [o.outputs[0].text for o in outputs]
            parsed = [srl.parse_output(t, r, s) for r, s in zip(raw, inputs)]
            result, gold_std = srl.align(t, inputs, parsed, gold)
            run = f"{key}_srl_{tag}"
            dump_json(raw, out_dir / f"{key}_srl_raw_{tag}.json")
            dump_json(result, out_dir / f"{key}_srl_result_{tag}.json")
            dump_json(gold_std, out_dir / f"{key}_srl_gold-std_{tag}.json")
            n_ok = sum(len(s["annotations"]) for s in result)
            print(f"\n=== {run}: {n_ok}/{len(inputs)} answers parsed, {len(result)} sentences ===")
            m = srl.score(t, result, gold_std)
            m.update(model=model, task=t, parsed=n_ok, queries=len(inputs))
            dump_json(m, out_dir / f"{run}.metrics.json")
            summary.append((key, t, m))
        del llm

    print("\n======== Summary ========")
    for key, t, m in summary:
        main_metric = {k: round(v, 4) for k, v in m.items() if k.endswith("f1")}
        print(f"{key} / {t}: {main_metric}")
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
