#!/usr/bin/env python
"""LoRA fine-tuning of Llama-3.1-8B-Instruct on the FrameNet instruction data.

This is the fine-tuning code used for the ACL 2026 model
(originally ``Fine_tuning_LLaMA_3_1_8b.py``). All hyperparameters now come
from the ``training`` section of the config; their values are the original
ones. The procedure is unchanged.

    python scripts/train.py --config configs/acl2026.yaml

Requirements: a CUDA GPU and the packages in requirements-train.txt
(Unsloth, and a TRL version that still provides
DataCollatorForCompletionOnlyLM, i.e. trl < 0.20).

The prompts are tokenized by the script itself (as SFTTrainer would do:
BOS added, truncation to max_seq_length, no padding) and the model is
trained with transformers' Trainer and DataCollatorForCompletionOnlyLM, so
the loss is computed only on the response tokens.

Data: by default the training set is read directly from Hugging Face,

    hf://datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions/train.json

(``training.train_data`` in the config). The file is downloaded once and
cached by the ``datasets`` library. ``--train-data`` accepts the same kind of
path, an https://huggingface.co/.../resolve/main/train.json URL, or a local
JSON/JSONL file. Only the fields Instruction, Question and Answer are used;
Task Family is ignored. No validation set is used: the model is trained for
the configured number of epochs, a checkpoint is saved every ``save_steps``
steps, and the final model is kept.

The script saves LoRA adapters. To use them with the vLLM evaluation
scripts, merge them into the base model with scripts/merge_lora.py.
"""

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama.config import cli_path, data_source, load_config, repo_path, section  # noqa: E402

# Alpaca-style instruction template (same as the original training script)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# Set in main() before training; module-level, as in the original script, so
# that the formatting function can be serialised by the datasets library.
EOS_TOKEN = ""
QUESTION_FIELD, ANSWER_FIELD, INSTRUCTION_FIELD = "Question", "Answer", "Instruction"


def formatting_prompts_func(examples):
    """
    Converts dataset rows into Alpaca-style prompts.
    Compatible with DataCollatorForCompletionOnlyLM.
    """
    questions = examples.get(QUESTION_FIELD, None)
    answers = examples.get(ANSWER_FIELD, None)
    inputs = examples.get(INSTRUCTION_FIELD, [""] * len(questions))  # optional field
    texts = []

    for inp, q, a in zip(inputs, questions, answers):
        prompt = ALPACA_PROMPT.format(inp, q, a) + EOS_TOKEN
        texts.append(prompt)

    return texts  # list, not a dict


def tokenize_batch(examples, tokenizer, max_seq_length):
    """Format a batch as Alpaca prompts and tokenize it (as SFTTrainer does)."""
    texts = formatting_prompts_func(examples)
    return tokenizer(texts, add_special_tokens=True, truncation=True,
                     max_length=max_seq_length, padding=False)


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning (ACL 2026 setup).")
    p.add_argument("--config", default="configs/acl2026.yaml")
    p.add_argument("--train-data", default=None,
                   help="Training data: hf://datasets/<repo>/<file>, an https URL, or a local file "
                        "(default: training.train_data)")
    p.add_argument("--output-dir", default=None, help="Override training.output_dir")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after N steps (for a quick smoke test only)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = section(load_config(args.config), "training")

    train_data = data_source(args.train_data or cfg["train_data"], from_cli=bool(args.train_data),
                             what="Training file")
    output_dir = cli_path(args.output_dir) if args.output_dir else repo_path(cfg["output_dir"])
    adapter_dir = repo_path(cfg.get("adapter_dir", output_dir / "final_adapter"))

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from unsloth import FastLanguageModel  # must be imported before transformers / trl
        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        sys.exit(f"ERROR: missing dependency ({exc}). Install requirements-train.txt.")
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        sys.exit("ERROR: this trl version has no DataCollatorForCompletionOnlyLM "
                 "(removed in trl 0.20). Install the versions in requirements-train.txt.")

    if not torch.cuda.is_available():
        sys.exit("ERROR: training needs a CUDA GPU.")

    seed = int(cfg.get("seed", 3407))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = cfg["base_model"]
    max_seq_length = int(cfg.get("max_seq_length", 2048))

    print(f"\nLoading {model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    global EOS_TOKEN, QUESTION_FIELD, ANSWER_FIELD, INSTRUCTION_FIELD
    EOS_TOKEN = tokenizer.eos_token

    lora = cfg.get("lora", {})
    print("Attaching LoRA adapters ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora.get("r", 8)),
        target_modules=list(lora.get("target_modules")),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.1)),
        bias=lora.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    QUESTION_FIELD = cfg.get("question_field", "Question")
    ANSWER_FIELD = cfg.get("answer_field", "Answer")
    INSTRUCTION_FIELD = cfg.get("instruction_field", "Instruction")

    print(f"\nLoading training data: {train_data}")
    try:
        dataset_train = load_dataset("json", data_files={"train": train_data})["train"]
    except Exception as exc:
        if train_data.startswith(("hf://", "http://", "https://")) and any(
                k in str(exc) for k in ("401", "403", "Unauthorized", "Repository Not Found", "gated")):
            sys.exit("ERROR: could not access the training data on Hugging Face. If the dataset "
                     f"is private, run: huggingface-cli login\nDetails: {exc}")
        raise
    print(f"  {len(dataset_train):,} examples")

    # ======================================================
    # Data collator: loss only on the response tokens
    # (same code as the original training script)
    # ======================================================
    response_template = "### Response:\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # Build one formatted example (first row)
    sample = formatting_prompts_func(dataset_train[:1])[0]
    print("----- SAMPLE PROMPT -----")
    print(sample[:800])
    print("Marker present? ->", "### Response:\n" in sample)
    print("-------------------------")

    marker = "### Response:\n"
    enc_full = tokenizer(sample, add_special_tokens=False).input_ids
    enc_marker = tokenizer(marker, add_special_tokens=False).input_ids

    def contains(sub, sup):
        # naive subsequence check
        L = len(sub)
        return any(sup[i:i + L] == sub for i in range(len(sup) - L + 1))

    print("Marker tokens found in prompt tokens? ->", contains(enc_marker, enc_full))

    # ======================================================
    # Tokenization (done here, in one process)
    # Same as SFTTrainer: prompt + EOS, BOS added by the tokenizer, truncated to
    # max_seq_length, no padding (the collator pads each batch). A fixed
    # fingerprint is given to datasets.map, so no function has to be pickled.
    # ======================================================
    print("\nTokenizing ...")
    fingerprint = "framellama-" + hashlib.sha256(
        f"{train_data}|{model_name}|{max_seq_length}|{len(dataset_train)}".encode()).hexdigest()[:16]
    dataset_tok = dataset_train.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        remove_columns=dataset_train.column_names,
        new_fingerprint=fingerprint,
        desc="Tokenizing",
    )
    print(f"  {len(dataset_tok):,} tokenized examples")

    t = cfg.get("args", {})
    training_args = TrainingArguments(
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
        warmup_steps=int(t.get("warmup_steps", 300)),
        num_train_epochs=float(t.get("num_train_epochs", 10)),
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=float(t.get("learning_rate", 5e-6)),
        fp16=True,
        bf16=False,
        logging_steps=int(t.get("logging_steps", 500)),
        save_steps=int(t.get("save_steps", 500)),
        eval_strategy="no",
        optim=t.get("optim", "adamw_8bit"),
        weight_decay=float(t.get("weight_decay", 0.01)),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        output_dir=str(output_dir),
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tok,
        data_collator=data_collator,
    )

    # Resume from the latest checkpoint in output_dir, if any
    checkpoint_dir = None
    if output_dir.is_dir():
        ckpts = sorted((d for d in output_dir.iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint-")),
                       key=lambda d: int(d.name.split("-")[-1]))
        if ckpts:
            checkpoint_dir = str(ckpts[-1])
            print(f"Resuming from checkpoint: {checkpoint_dir}")

    print("\nStarting training ...")
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapters saved to {adapter_dir}")
    print("Merge them for vLLM with: python scripts/merge_lora.py --config", args.config)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
