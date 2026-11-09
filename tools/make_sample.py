"""Pick a few records per task from a generated file to make data/generated/sample.jsonl.

    python scripts/generate_dataset.py --frames Killing --jsonl data/generated/killing.jsonl
    python tools/make_sample.py data/generated/killing.jsonl data/generated/sample.jsonl --per-task 2
"""

import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--per-task", type=int, default=2)
    args = p.parse_args()

    picked = defaultdict(list)
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if len(picked[rec["task_id"] or rec["task_key"]]) < args.per_task:
                picked[rec["task_id"] or rec["task_key"]].append(line)
    with open(args.output, "w", encoding="utf-8") as f:
        for key in picked:
            f.writelines(picked[key])
    print(f"Wrote {sum(map(len, picked.values()))} records ({len(picked)} task keys) to {args.output}")


if __name__ == "__main__":
    main()
