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
import json
import re
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
# ======================================================
# ✅ Robust JSON Extraction & Parsing (LLM-safe)
# ======================================================

def unwrap_outer_quotes(text):
    """
    If the entire output is wrapped as a quoted string,
    unwrap it safely (handles escaped quotes).
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    if text.startswith('"') and text.endswith('"'):
        try:
            # Converts \" → "
            return json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1].strip()

    return text


def extract_first_json_object(text):
    """
    Extract the first balanced JSON object {...}
    ignoring any text before it.
    """
    if not isinstance(text, str):
        return ""

    start = text.find("{")
    if start == -1:
        return ""

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i + 1]

    return ""


def sanitize_json_text(text):
    """
    Light cleanup without altering structure.
    """
    text = text.replace("\\n", " ").replace("\n", " ")
    text = text.replace("\\t", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text


import json

def robust_json_parser(raw):
    """
    Safely extract and parse a JSON object from model output.
    Keeps raw text intact if parsing fails.
    """

    # -------------------------
    # Step 1: Ensure string
    # -------------------------
    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str):
        raise ValueError("Input is not string or dict")

    text = raw.strip()

    # -------------------------
    # Step 2: If fully quoted, unwrap ONCE
    # -------------------------
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)  # removes outer quotes + unescapes
        except json.JSONDecodeError:
            pass  # leave as-is

    # -------------------------
    # Step 3: Trim before first {
    # -------------------------
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    text = text[start:]

    # -------------------------
    # Step 4: Trim after last }
    # -------------------------
    end = text.rfind("}")
    if end == -1:
        raise ValueError("No JSON object found")

    text = text[:end + 1]

    # -------------------------
    # Step 5: Parse JSON
    # -------------------------
    return json.loads(text)


import json
import re

def safe_parse_json(raw_text):
    """
    Safely extract and parse a JSON object from malformed LLM output.
    If parsing fails, return the original raw_text.
    """

    if raw_text is None:
        return raw_text

    text = raw_text.strip()

    # --------------------------------------------------
    # STEP 1: If text is quoted, UNQUOTE IT FIRST
    # --------------------------------------------------
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except Exception:
            # cannot unwrap → keep going with original
            text = text[1:-1]

    # --------------------------------------------------
    # STEP 2: Remove everything BEFORE first '{'
    # --------------------------------------------------
    first_brace = text.find("{")
    if first_brace == -1:
        return raw_text  # no JSON possible

    text = text[first_brace:].strip()

    # --------------------------------------------------
    # STEP 3: Extract EXACT balanced JSON block
    # --------------------------------------------------
    stack = []
    buf = ""

    for ch in text:
        if ch == "{":
            stack.append("{")
        if stack:
            buf += ch
        if ch == "}":
            stack.pop()
            if not stack:
                break

    # --------------------------------------------------
    # STEP 4: Final parse attempt
    # --------------------------------------------------
    try:
        return json.loads(buf)
    except Exception:
        return raw_text


# ======================================================
# ✅ Frame-based Filtering (unchanged logic)
# ======================================================

def filter_json_by_frame(json_objects, target_frame):
    """
    Filters parsed JSON objects by target frame.
    """
    filtered = []
    seen = set()

    for obj in json_objects:
        for ann in obj.get("annotations", []):
            frame = ann.get("frame")
            if frame == target_frame and frame not in seen:
                filtered.append(obj)
                seen.add(frame)
                break

    return filtered




def build_prompt(frame_combine, lexical_combine, frame, sentence, lexical_unit):
    return f"""
You are an SRL system trained on FrameNet.
Your task is to extract the frame elements expressed by the lexical unit.
and return their exact spans from the sentence.

FOCUS ON TWO REQUIREMENTS:

A) FRAME ELEMENT IDENTIFICATION
   - Extract ONLY the frame elements that belong to the given frame.
   - A frame element(FE) must be explicitly expressed in the text.
   - Ignore any FE that is not evoked by this lexical unit in this sentence.
   - Do NOT invent or hallucinate FE.

B) PRECISE SPAN MAPPING
   - For every FE you extract, return the exact substring from the sentence.
   - Do NOT paraphrase, shorten, or expand spans.
   - The FE span must come from the same clause or phrase where the lexical unit appears.

OUTPUT CONSTRAINT (STRICT):
- You MUST output a single valid JSON object.
- You MUST NOT output any text before '{' and after with '}' even single letter is not allowed.
- You MUST NOT repeat the JSON.
- You MUST NOT include explanations, comments, examples, or markdown.

READ-ONLY CONTEXT (DO NOT OUTPUT)
{{
  "input_sentence": "{sentence}",
  "frames_in_sentence": "{frame_combine}",
  "lexical_units_in_sentence": "{lexical_combine}"
}}

PURPOSE OF THIS CONTEXT:
The sentence may evoke multiple frames in different clauses or phrases.
Use this context only to locate the correct evoked region for the TARGET
lexical unit and avoid extracting roles from other events.

NOW PROCESS THE REAL JSON INPUT:
{{
  "input_sentence": "{sentence}",
  "target_frame": "{frame}",
  "target_lexical_unit": "{lexical_unit}"
}}

COMPLETE THE JSON OBJECT AS PER THE FORMAT BELOW:
{{
  "input_sentence": "{sentence}",
  "annotations": [
    {{
      "frame": "{frame}",
      "lexical_unit": "{lexical_unit}",
      "frame_elements": {{
        "<FrameElement>": "<Exact span from the sentence>"
        "<FrameElement>": "<Exact span from the sentence>"

      }}
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
        frame_combine = ", ".join(sample['frame_combine'])
        lexical_combine = ", ".join(sample['lexical_combine'])
        prompt = build_prompt(frame_combine, lexical_combine, frame, sentence, lexical_unit)
        #prompt = build_prompt(frame_combine, lexical_combine, frame, sentence, lexical_unit)

        prompts.append(prompt)
        frames.append(frame)

    # Generate all outputs with vLLMs
    outputs = llm.generate(prompts, sampling_params)

    # Process outputs
    for i, result in enumerate(tqdm(outputs, desc="Parsing Outputs")):

        raw_text = result.outputs[0].text
        frame = frames[i]  # frame alignment preserved

        try:
            parsed = safe_parse_json(raw_text)

            results.append(parsed)

        except Exception:
            # 🔴 IMPORTANT: keep raw text exactly as produced by model
            results.append(raw_text)

    return results


# ======================================================
# ✅ Main function (required for vLLM)
# ======================================================
def main():
    MODEL_PATH = "merged_LLama_SRL_full"
    #MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct" 
    INPUT_FILE = "Evaluation/SRL/input_sentences_combined.json"
    OUTPUT_FILE = "Evaluation/Pred/SRL/Pred/adapted_srl_fe_old.json"


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
