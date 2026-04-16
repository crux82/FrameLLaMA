#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast SRL/FrameNet JSON generation using vLLM instead of Transformers.
Converts Unsloth fine-tuned or merged model (FP16) for ultra-fast inference.
"""

import os
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ======================================================
# ✅ JSON & Cleaning Helpers
# ======================================================
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_blocks(text):
    json_blocks, stack, buf = [], [], ""
    inside = False

    for c in text:
        if c == "{":
            inside = True
            stack.append("{")
            buf += c
        elif c == "}":
            if stack:
                stack.pop()
                buf += c
                if not stack:
                    json_blocks.append(buf)
                    buf, inside = "", False
        elif inside:
            buf += c
    return json_blocks


def remove_newlines_from_output(raw_output):
    cleaned = []
    for item in raw_output:
        if isinstance(item, list) and len(item) == 1:
            item = item[0]
        if isinstance(item, str):
            cleaned_item = item.replace("\\n", " ").replace("\n", " ").replace("\\t", " ").strip()
            cleaned.append(cleaned_item)
    return cleaned


def sanitize_json_text(text):
    text = text.replace("\\n", " ").replace("\n", " ").replace("\\t", " ").replace('"{', '[{').replace('}"', '}]').strip()
    text = re.sub(r'\[" "]', '', text)
    text = re.sub(r'"([A-Za-z0-9_\- ]+):[^"]*"\s*:', lambda m: f'"{m.group(1).strip()}":', text)
    text = re.sub(r'"([A-Za-z0-9_\- ]+): ([^"]+)"', lambda m: f'"{m.group(1).strip()}": "{m.group(2).strip()}"', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def filter_json_by_frame(json_blocks, target_frame):
    filtered = []
    seen = set()
    for block in json_blocks:
        try:
            cleaned = sanitize_json_text(block)
            parsed = json.loads(cleaned)
            for ann in parsed.get("annotations", []):
                frame = ann.get("frame")
                if frame == target_frame and frame not in seen:
                    filtered.append(parsed)
                    seen.add(frame)
                    break
        except json.JSONDecodeError:
            continue
    return filtered


def build_prompt(frame, sentence, lexical_unit):
    return f"""
You are an expert in Frame Semantics and FrameNet.
Your task is to determine which FrameNet frame is evoked by the given lexical unit in the sentence.

INSTRUCTIONS:
1. Analyze the provided sentence.
2. Focus ONLY on the given lexical unit as the target of analysis.
3. Identify the FrameNet frame that this lexical unit evokes in this specific sentence.
4. Do NOT guess frames unrelated to the lexical unit.
5. Do NOT output definitions, explanations, or extra text.
6. The frame MUST be a valid FrameNet frame name.
7. Output strictly valid JSON.

EXAMPLE
Lexical Unit: trip
Sentence: Hall recently returned from a trip to Iraq.

Expected Output:
{{
  "input_sentence": "Hall recently returned from a trip to Iraq.",
  "annotations": [
    {{
      "lexical_unit": "trip",
      "frame": "Travel"
    }}
  ]
}}

NOW PROCESS THE REAL INPUT:

LEXICAL UNIT: {lexical_unit}
SENTENCE: {sentence}

REQUIRED JSON FORMAT:
{{
  "input_sentence": "{sentence}",
  "annotations": [
    {{
      "lexical_unit": "{lexical_unit}",
      "frame": "<Predicted FrameNet frame name>"
    }}
  ]
}}"""



def generate_predictions_vllm(llm, sampling_params, samples):
    results = []
    prompts = []
    frames = []  # store frames aligned with prompts

    # Build prompts
    for sample in tqdm(samples, desc="Building Prompts"):
        frame = sample['frame'].replace("_", " ")
        lexical_unit = sample['lexical_unit']
        sentence = sample['sentence']
        prompt = build_prompt(frame, sentence, lexical_unit)

        prompts.append(prompt)
        frames.append(frame)

    # Generate all outputs with vLLM
    outputs = llm.generate(prompts, sampling_params)

    # Process outputs
    for i, result in enumerate(tqdm(outputs, desc="Parsing Outputs")):

        raw_text = result.outputs[0].text.strip()
        frame = frames[i]  # correct frame per sample

        # Extract JSON-like blocks
        json_blocks = extract_json_blocks(raw_text)
        cleaned_blocks = remove_newlines_from_output(json_blocks)
        filtered = filter_json_by_frame(cleaned_blocks, frame)

        results.append(filtered if filtered else raw_text)

    return results




# ======================================================
# ✅ Main function (required for vLLM)
# ======================================================
def main():
    MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    INPUT_FILE = "Evaluation/SRL/input_sentences.json"
    OUTPUT_FILE = "Evaluation/Pred/SRL/baseline_frame.json"


    print("📄 Loading samples...")
    samples = load_json(INPUT_FILE)
    print(f"✅ Loaded {len(samples)} SRL samples.")

    print("⚡ Starting vLLM engine...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["```","</s>"]

    )

    print("🚀 Generating predictions...")
    predictions = generate_predictions_vllm(llm, sampling_params, samples)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved predictions to: {OUTPUT_FILE}")


# ======================================================
# 🔒 Entry Point
# ======================================================
if __name__ == "__main__":
    main()
