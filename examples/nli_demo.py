#!/usr/bin/env python
"""Run FrameLLaMA on one natural language inference example.

    python examples/nli_demo.py

    # your own example
    python examples/nli_demo.py --premise "Alice sold a bicycle to Bob." \\
        --hypothesis "Bob bought a bicycle from Alice." --expected entailment

    # show the prompt without loading the model (no GPU needed)
    python examples/nli_demo.py --dry-run

The prompt and decoding are the ones used for the zero-shot NLI evaluation in
the paper (evaluation/prompts/nli_zero_shot_confer.txt, greedy decoding,
2 new tokens). The model was fine-tuned on Alpaca-style prompts and evaluated
with plain-text prompts, so no chat template is applied by default; add
--chat to wrap the same prompt in the Llama 3.1 chat template instead.

Requirements: Linux, an NVIDIA GPU with about 18 GB of free memory, and the
packages in requirements.txt.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama import nli  # noqa: E402

MODEL = "sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17"

PREMISE = "Alice sold a bicycle to Bob."
HYPOTHESIS = "Bob bought a bicycle from Alice."
EXPECTED = "entailment"


def parse_args():
    p = argparse.ArgumentParser(description="FrameLLaMA NLI demonstration.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model", default=MODEL, help="Hugging Face id or local folder")
    p.add_argument("--base-model", default=None, help="Base model, if --model holds LoRA adapters")
    p.add_argument("--premise", default=PREMISE)
    p.add_argument("--hypothesis", default=HYPOTHESIS)
    p.add_argument("--expected", default=None,
                   help="Expected label (entailment/contradiction/neutral or E/C/N), shown for reference")
    p.add_argument("--chat", action="store_true", help="Apply the model's chat template to the prompt")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true", help="Print the prompt and exit")
    args = p.parse_args()
    if args.expected is None and args.premise == PREMISE and args.hypothesis == HYPOTHESIS:
        args.expected = EXPECTED
    return args


def main():
    args = parse_args()
    template = nli.read_prompt(nli.PROMPT_FILES[("zero-shot", "confer")])
    prompt = nli.fill(template, args.premise.strip(), args.hypothesis.strip())

    print("=" * 60)
    print("FrameLLaMA NLI demonstration")
    print("=" * 60)
    print(f"Model: {args.model}")

    if args.dry_run:
        print("\nPrompt sent to the model:\n" + "-" * 60 + f"\n{prompt}\n" + "-" * 60)
        print("Decoding:", nli.sampling_kwargs("zero-shot"))
        return

    from framellama.vllm_utils import generate, load_llm
    from vllm import SamplingParams

    print("\nLoading the model (the first run downloads about 16 GB) ...", flush=True)
    llm, lora = load_llm(args.model, gpu_memory_utilization=args.gpu_memory_utilization,
                         max_model_len=args.max_model_len, base_model=args.base_model)

    if args.chat:
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)

    params = SamplingParams(**nli.sampling_kwargs("zero-shot"))
    raw = generate(llm, [prompt], params, lora)[0].outputs[0].text
    predicted = nli.normalize_model_output(raw)

    print("\nPremise:")
    print(args.premise)
    print("\nHypothesis:")
    print(args.hypothesis)
    print(f"\nRaw model output: {raw.strip()!r}")
    if not raw.strip() or raw.strip()[0].upper() not in "ECN":
        print("(The output is not one of E, C, N; the label below comes from the fallback rules.)")
    print(f"\nPredicted label: {predicted.capitalize()}")
    if args.expected:
        print(f"Expected label:  {nli.normalize_reference_label(args.expected).capitalize()}")
    print("=" * 60)
    print("This is a single example, not a reproduction of the paper's results.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
