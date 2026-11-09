#!/usr/bin/env python
"""Generate the FrameNet instruction-tuning data, one CSV file per frame.

FrameNet 1.7 and WordNet are read through NLTK and downloaded automatically
the first time (as in the original script), so no other setup is needed.

Examples
--------
One frame:

    python scripts/generate_dataset.py --frames Killing

Several frames:

    python scripts/generate_dataset.py --frames Commerce_sell Commerce_buy Arriving

All 1,221 FrameNet 1.7 frames (the full corpus):

    python scripts/generate_dataset.py

Output: data/generated/<Frame name>.csv for every frame, in the format of
the original research script (columns: Frame Name, Prompt Type, Question ID,
Question Sample ID, Answer Type, Answer Sample ID, Questions, Answers).
Frames whose CSV file already exists are skipped, so an interrupted run can
simply be started again (use --overwrite to regenerate them). A summary with
the number of examples per frame and per task is written to
<output-dir>/_summary.json.

Optionally, --jsonl FILE also writes all examples to one JSON-lines file with
the field names of the released dataset (Question, Answer, Task Family) plus
provenance fields (paper task number, frame, templates); see data/README.md.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama.config import cli_path, load_config, section  # noqa: E402

SOURCE = "FrameNet 1.7"


def ensure_hash_seed():
    """Set PYTHONHASHSEED=0 (re-starting Python if needed) so set ordering is fixed."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate FrameNet instruction-tuning data (one CSV per frame).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--frames", nargs="+", default=None,
                   help="FrameNet frame names, e.g. Killing Commerce_sell (default: the frames in "
                        "the config, i.e. all frames)")
    p.add_argument("--output-dir", default=str(ROOT / "data" / "generated"),
                   help="Folder for the per-frame CSV files (default: data/generated)")
    p.add_argument("--jsonl", default=None,
                   help="Also write all examples to this JSON-lines file")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate frames whose CSV file already exists")
    p.add_argument("--config", default="configs/acl2026.yaml",
                   help="Settings: seed, embedding model, negative threshold (default: configs/acl2026.yaml)")
    p.add_argument("--seed", type=int, default=None, help="Override the seed from the config")
    p.add_argument("--max-frames", type=int, default=None, help="Generate only the first N selected frames")
    p.add_argument("--verbose", action="store_true", help="Show the generator's per-frame printout")
    p.add_argument("--framenet-dir", default=None, help=argparse.SUPPRESS)  # optional local FrameNet copy
    return p.parse_args()


def task_key(prompt_type, question_id):
    return f"{prompt_type} | Q{question_id}"


def csv_name(frame):
    """File name used by the original script: underscores replaced by spaces."""
    return f"{frame.replace('_', ' ')}.csv"


def row_to_record(row, task_families):
    """One generated row as a JSON record (used for --jsonl).

    Question, Answer and Task Family have the names and meaning of the released
    training data (Task Family = the generator's prompt type, e.g.
    "Frame Definition"). The other fields record provenance: the paper's task
    number (T1-T40), the frame, and the question/answer templates used.
    """
    frame, prompt_type, qid, q_sample, a_type, a_sample, question, answer = row
    key = task_key(prompt_type, qid)
    return {
        "Question": question,
        "Answer": answer,
        "Task Family": prompt_type,
        "task_id": task_families.get(key),
        "task_key": key,
        "frame_name": frame,
        "question_id": str(qid),
        "question_template": q_sample,
        "answer_type": a_type,
        "answer_template": a_sample,
        "source": SOURCE,
    }


def main():
    args = parse_args()
    ensure_hash_seed()

    cfg = load_config(args.config)
    gen = section(cfg, "generation")
    seed = args.seed if args.seed is not None else gen.get("seed")
    max_frames = args.max_frames if args.max_frames is not None else gen.get("max_frames")
    frames_cfg = args.frames or gen.get("frames", "all")
    embedding_model = gen.get("embedding_model", "all-MiniLM-L6-v2")
    negative_threshold = float(gen.get("negative_threshold", 0.50))
    from framellama.task_families import TASK_FAMILIES as task_families

    out_dir = cli_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from framellama import dataset_builder as builder

    print("[1/3] Loading FrameNet 1.7 (NLTK; downloaded automatically the first time), "
          f"lookup tables and embedding model ({embedding_model}) ...", flush=True)
    try:
        fn = builder.setup(
            framenet_dir=str(cli_path(args.framenet_dir)) if args.framenet_dir else None,
            download=True,
            embedding_model=embedding_model,
            negative_threshold=negative_threshold,
            csv_dir=str(out_dir),
            show_progress=True,
        )
    except (FileNotFoundError, LookupError, ImportError) as exc:
        sys.exit(f"ERROR: {exc}")

    all_names = [f.name for f in fn.frames()]
    if frames_cfg in (None, "all"):
        selected = all_names
    else:
        known = set(all_names)
        wanted = [str(n).replace(" ", "_") for n in frames_cfg]
        unknown = [n for n in wanted if n not in known]
        if unknown:
            sys.exit(f"ERROR: unknown FrameNet frame(s): {', '.join(unknown)}")
        selected = wanted
    if max_frames:
        selected = selected[: int(max_frames)]

    existing = [] if args.overwrite else [n for n in selected if (out_dir / csv_name(n)).exists()]
    todo = [n for n in selected if n not in set(existing)]
    print(f"[2/3] Generating {len(todo)} frame(s) (seed={seed}); "
          f"{len(existing)} already present in {out_dir} and skipped.", flush=True)

    counts = Counter()
    frame_counts = {}
    failed = {}
    n_records = 0
    start = time.time()
    jsonl = None
    if args.jsonl:
        cli_path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl = open(cli_path(args.jsonl), "w", encoding="utf-8")

    try:
        for i, name in enumerate(todo, 1):
            try:
                if args.verbose:
                    rows = builder.generate_frame(name, seed=seed)
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = builder.generate_frame(name, seed=seed)
            except Exception as exc:  # the original script also skipped such frames
                failed[name] = f"{type(exc).__name__}: {exc}"
                print(f"  ! {name}: skipped ({failed[name]})", flush=True)
                continue

            if rows:
                with open(out_dir / csv_name(name), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(builder.CSV_COLUMNS)
                    w.writerows(rows)

            for row in rows:
                counts[task_key(row[1], row[2])] += 1
                if jsonl:
                    jsonl.write(json.dumps(row_to_record(row, task_families), ensure_ascii=False) + "\n")
            n_records += len(rows)
            frame_counts[name] = len(rows)
            print(f"  {i}/{len(todo)}  {csv_name(name):45s} {len(rows):6,d} examples  "
                  f"({time.time() - start:.0f}s)", flush=True)
    finally:
        if jsonl:
            jsonl.close()

    summary = {
        "output_dir": str(out_dir),
        "num_examples": n_records,
        "num_frames_generated": sum(1 for v in frame_counts.values() if v),
        "num_frames_skipped_existing": len(existing),
        "examples_per_frame": frame_counts,
        "examples_per_task": {f"{task_families.get(k) or '?'}  {k}": v for k, v in sorted(counts.items())},
        "failed_frames": failed,
        "settings": {
            "seed": seed,
            "embedding_model": embedding_model,
            "negative_threshold": negative_threshold,
            "source": SOURCE,
        },
        "environment": {
            "python": platform.python_version(),
            "nltk": __import__("nltk").__version__,
        },
    }
    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[3/3] Wrote {n_records:,} examples for {len(frame_counts)} frame(s) to {out_dir}")
    if args.jsonl:
        print(f"      JSON lines: {cli_path(args.jsonl)}")
    if failed:
        print(f"      {len(failed)} frame(s) failed; see 'failed_frames' in _summary.json.")


if __name__ == "__main__":
    main()
