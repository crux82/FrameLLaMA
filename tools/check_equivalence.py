"""Check that src/framellama generates exactly what the original research script generates.

Runs tools/original/framenet_dataset_generator.py (patched only to write to a
temporary folder, to process the chosen frames, and to seed each frame like
the new code) and scripts/generate_dataset.py on the same frames, then
compares the per-frame CSV files byte by byte.

    python tools/check_equivalence.py --frames Commerce_sell Killing Arriving

Needs NLTK framenet_v17 + wordnet and sentence-transformers. Each side loads
all of FrameNet first, which takes a few minutes.
"""

import argparse
import filecmp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "tools" / "original" / "framenet_dataset_generator.py"


def patch_original(src, out_dir, frames, seed):
    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, f"patch point not found exactly once: {old!r}"
        src = src.replace(old, new)
    rep('target_folder = "Dataset_1/Train/English_1"', f'target_folder = {str(out_dir)!r}')
    rep('    process_frames = fn.frames()', f'    process_frames = [fn.frame_by_name(n) for n in {frames!r}]')
    rep('        frame_name = item["frame_name"]\n',
        '        frame_name = item["frame_name"]\n'
        f'        random.seed(f"{seed}:{{frame_name}}")  # added for the equivalence check\n')
    return src


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", nargs="+", default=["Commerce_sell", "Abandonment", "Killing"])
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    env = dict(os.environ, PYTHONHASHSEED="0")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        orig_out, new_out = tmp / "original", tmp / "new"
        orig_out.mkdir()
        patched = tmp / "original_patched.py"
        patched.write_text(patch_original(ORIGINAL.read_text(encoding="utf-8"), orig_out, args.frames, args.seed),
                           encoding="utf-8")

        print("Running the original script ...", flush=True)
        subprocess.run([sys.executable, str(patched)], check=True, cwd=tmp, env=env,
                       stdout=subprocess.DEVNULL)
        print("Running scripts/generate_dataset.py ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_dataset.py"),
                        "--frames", *args.frames, "--seed", str(args.seed),
                        "--output-dir", str(new_out)], check=True, cwd=tmp, env=env)

        ok = True
        for f in sorted(orig_out.glob("*.csv")):
            same = (new_out / f.name).exists() and filecmp.cmp(f, new_out / f.name, shallow=False)
            ok &= same
            print(f"{'IDENTICAL' if same else 'DIFFERENT'}  {f.name}")
        if not list(orig_out.glob("*.csv")):
            ok = False
            print("The original script produced no files.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
