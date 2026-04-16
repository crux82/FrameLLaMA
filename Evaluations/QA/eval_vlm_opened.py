#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned FP16 model (merged LoRA + base) using vLLM.
This script:
  - Loads a test dataset in JSON format
  - Builds prompts in Alpaca style
  - Generates predictions via vLLM
  - Extracts Yes/No (or text) answers
  - Saves results to CSV for evaluation
"""

import os, csv, re, string
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ======================================================
# ✅ Alpaca-style prompt format (same as training)
# ======================================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# ======================================================
# ✅ Build prompt from dataset
# ======================================================
def build_prompt(sample):
    instruction = sample.get("Instruction", "")
    question = sample.get("Question", "")
    return alpaca_prompt.format(instruction, question)

# ======================================================
# ✅ Extract Yes/No (or general text)
# ======================================================
def extract_final_answer(text, prompt="", valid_answers=["Yes", "No"]):
    """
    Extracts the Yes/No answer from model output, or returns the cleaned text.
    """
    cleaned_text = text.replace(prompt, "").strip()
    tokens = re.split(r'\s|\n', cleaned_text)
    for tok in reversed(tokens):
        tok_clean = tok.strip().capitalize().strip(string.punctuation)
        if tok_clean in valid_answers:
            return tok_clean
    # Fallback: return cleaned text if no Yes/No found
    return cleaned_text

# ======================================================
# ✅ Write results to CSV
# ======================================================
def write_to_csv(rows, file_name="Evaluation/Pred/QA/finetuned_mcq.csv"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question","Predictions", "References"])
        writer.writerows(rows)
    print(f"✅ Results saved to {file_name}")

# ======================================================
# ✅ Generate predictions with vLLM
# ======================================================
def generate_predictions(llm, dataset):
    rows = []
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

    for i, sample in enumerate(dataset):
        prompt = build_prompt(sample)
        outputs = llm.generate([prompt], sampling_params)
        
        raw_output = outputs[0].outputs[0].text.strip()

        reference = sample.get("Answer", "").strip()

        rows.append([sample["Question"], raw_output, reference])

        if i % 100 == 0:
            print(f"[{i}/{len(dataset)}] ✅ Processed")

    return rows

# ======================================================
# ✅ MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    # Load dataset
    DATA_PATH = "Evaluation/JSON/all_test_mcq.json"
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    print(f"📊 Loaded dataset: {len(dataset)} samples")

    # Model path
    MODEL_PATH = "merged_finetuned_llama3.1_8b_fp16_vllm"

    # Initialize vLLM
    print("⚡ Starting vLLM engine...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        disable_log_stats=True,
    )

    # Run inference
    print("🚀 Generating predictions...")
    results = generate_predictions(llm, dataset)

    # Save results
    write_to_csv(results)

    print("🎉 Evaluation completed successfully!")
