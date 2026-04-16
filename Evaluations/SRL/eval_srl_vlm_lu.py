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
You are trained on FrameNet.
Your task is to extract ONLY the lexical unit that are expressed
in the part of the sentence where the given frame evokes.

RULES:
1. Identify the phrase or clause in which the frame appears.
2. Extract only the lexical unit present in this frame-evoked region.
3. Do NOT extract lexical unit from other parts of the sentence.
4. Do NOT invent lexical unit. Use only those belonging to the given frame.
5. Output strictly valid JSON.

Expected OUTPUT:
{{
  "input_sentence": "Hall, who recently returned from a trip to Iraq, said sanctions hurt civilians.",
  "annotations": [
          {{
			  "frame": "Location",
			  "lexical unit": "trip"
		   }}
      ]
}}

NOW PROCESS THE REAL INPUT:

FRAME: {frame}
SENTENCE: {sentence}

REQUIRED JSON FORMAT:
{{
  "input_sentence": "{sentence}",
  "annotations": [
          {{
			  "frame":"{frame}"
			   "lexical_unit": "<Exact lexical unit from frame-evoked part>"
		  }}
      ]
}}
        
        """



def generate_predictions_vllm(llm, sampling_params, samples):
    results = []
    prompts = []
    frames = []  # store frames aligned with prompts

    # Build prompts
    for sample in tqdm(samples, desc="Building Prompts"):
        frame = sample['frame'].replace("_", " ")
        lexical_unit = sample['lexical_unit']
        sentence = sample['sentence']
        frame_combine = ", ".join(sample['frame_combine'])
        lexical_combine = ", ".join(sample['lexical_combine'])
        #prompt = build_prompt_2(frame_combine, lexical_combine, frame, sentence, lexical_unit)
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
    MODEL_PATH = "merged_LLama_SRL_full"
    #MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct" 
    INPUT_FILE = "Evaluation/SRL/input_sentences_combined_1.json"
    OUTPUT_FILE = "Evaluation/Pred/SRL/Pred/adapted_srl_lu_old.json"


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
