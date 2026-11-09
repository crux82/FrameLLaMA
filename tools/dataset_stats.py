"""Statistics of the released instruction-tuning files, for the dataset card.

    python tools/dataset_stats.py data/hf/train.json data/hf/test/*.json

For each file: number of examples, field check, number of examples per
question type and per Task Family, and the SHA-256 checksum. The Markdown tables it prints can be
pasted into the dataset card (docs/huggingface/dataset_card.md) and the
numbers into data/manifests/acl2026.json. Works with a JSON list or JSONL.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framellama.qa import answer_type  # noqa: E402

FIELDS = ["Instruction", "Question", "Answer", "Task Family"]


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    families = {}
    types = {}
    totals = {}
    print("## Files\n")
    print("| Split | File | Examples | SHA-256 |")
    print("|---|---|---|---|")
    for path in args.files:
        data = load(path)
        split = Path(path).stem
        missing = Counter(f for rec in data for f in FIELDS if f not in rec)
        empty = Counter(f for rec in data for f in ("Question", "Answer") if not str(rec.get(f, "")).strip())
        extra = Counter(k for rec in data for k in rec if k not in FIELDS)
        totals[split] = len(data)
        families[split] = Counter(rec.get("Task Family", "(none)") for rec in data)
        types[split] = Counter(answer_type(rec.get("Answer")) for rec in data)
        print(f"| {split} | `{Path(path).name}` | {len(data):,} | `{sha256(path)}` |")
        for name, c in (("missing fields", missing), ("empty Question/Answer", empty), ("other fields", extra)):
            if c:
                print(f"\n> {path}: {name}: {dict(c)}\n")
    print(f"| **total** | | **{sum(totals.values()):,}** | |")

    splits = list(families)
    print("\n## Examples per question type\n")
    print("| Question type | " + " | ".join(splits) + " |")
    print("|---|" + "---|" * len(splits))
    for t, label in (("opened", "Open-ended"), ("closed", "Yes/No"), ("mcq", "Multiple choice")):
        print(f"| {label} | " + " | ".join(f"{types[s][t]:,}" for s in splits) + " |")

    all_fams = sorted(set().union(*families.values()), key=lambda f: -sum(families[s][f] for s in splits))
    print("\n## Examples per Task Family\n")
    print("| Task Family | " + " | ".join(splits) + " |")
    print("|---|" + "---|" * len(splits))
    for fam in all_fams:
        print(f"| {fam} | " + " | ".join(f"{families[s][fam]:,}" for s in splits) + " |")


if __name__ == "__main__":
    main()
