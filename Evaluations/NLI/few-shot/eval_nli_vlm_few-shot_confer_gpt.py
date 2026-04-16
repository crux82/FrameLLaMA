
import os
import csv
import re
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
from datetime import date

# vLLM + transformers
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Optional SBERT (comment out if not installed)
try:
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# ================================
# CONFIG
# ================================
#MODEL_PATH = "merged_finetuned_llama3.1_8b_fp16_vllm_full"   # change as needed
#MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"   # change as needed
#MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#MODEL_PATH = "google/gemma-3-4b-it"
MODEL_PATH = "openai/gpt-oss-20b"
TEST_FILE = "Evaluation/NLI/snli_matched_FrameNet.json"               # change as sneeded (json list)
OUTPUT_CSV = "Evaluation/Pred/NLI/Pre/baseline_nli_0shot_gpt.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def compute_sbert_similarity(references: List[str], predictions: List[str]) -> float:
    if not SBERT_AVAILABLE:
        return float("nan")
    model = SentenceTransformer("all-MiniLM-L12-v2")
    emb_ref = model.encode(references, convert_to_tensor=True)
    emb_pred = model.encode(predictions, convert_to_tensor=True)
    sims = []
    for r, p in zip(emb_ref, emb_pred):
        sims.append(1 - cosine(r.cpu().numpy(), p.cpu().numpy()))
    return float(np.mean(sims))

# ================================
# PROMPT (0-shot, safe)
# ================================
current_date_str = date.today().isoformat()
SYSTEM_MESSAGE = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {current_date_str}
Reasoning: medium
# Valid channels: analysis, commentary, final. Channel must be included for every message.
<|end|>"""

DEVELOPER_MESSAGE = """<|start|>developer<|message|># Instructions
You are a strict logical evaluator. Your task is to judge if the hypothesis necessarily follows from the premise.
Consider the truth of the premise. If the premise is true, determine if the hypothesis must also be true.
Output exactly ONE of the following **and only the following**:
   - assistantfinalE: if the hypothesis MUST always be true
   - assistantfinalC: if the hypothesis MUST always be false
   - assistantfinalN: if the hypothesis MAY be true OR false

Rules:
1. Output exactly ONE token: assistantfinalE, assistantfinalC, or assistantfinalN
2. Do NOT output reasoning, explanations, punctuation, quotes, labels, or any extra text
3. Output exactly ONE line

### Examples:
    Premise: If Emily passes her driving test, she'll never read a book by Dan Brown again.  
    Hypothesis: Emily has read a book by Dan Brown before.  
    Output: assistantfinalE

    Premise: If Margaret's flight arrives on time, her friend will be waiting for her at the airport.  
    Hypothesis: Margaret has a friend.  
    Output: assistantfinalE

    Premise: Sam believes that if the boss rewarded a good behavior by Nava, they'll never reward a good behavior by Nava again.  
    Hypothesis: The boss rewarded a good behavior by Nava.  
    Output: assistantfinalN

    Premise: If Alexander has a bright roommate, he'll rest with his bright roommate.  
    Hypothesis: Alexander has a bright roommate.  
    Output: assistantfinalN

    Premise: It's not the case that if Curtis reports on this issue, he'll never report on an issue again.  
    Hypothesis: Curtis has reported on an issue before.  
    Output: assistantfinalN

    Premise: Sam believes that if Alice ever drank a cocktail, she didn't drink martini again.  
    Hypothesis: Alice has drunk a martini.  
    Output: assistantfinalN

    Premise: If Meredith has video games, she will play her immersive video game.  
    Hypothesis: Meredith has an immersive video game.  
    Output: assistantfinalN <|end|>"""


# ================================
# MAIN: generate predictions
# ================================
def generate_predictions_vllm(llm: LLM, dataset: List[dict]) -> Tuple[List[str], List[str]]:
    """
    dataset: list of dicts with keys 'premise','hypothesis','gold_label' (gold_label optional)
    returns: predictions (normalized to SNLI labels), references (normalized)
    """
    prompts = []
    refs = []
    rows_for_csv = []

    for item in dataset:
        p = item.get("premise", "").strip()
        h = item.get("hypothesis", "").strip()
        refs.append(item.get("gold_label", ""))
         # --- Prepare prompts ---
        prompt = SYSTEM_MESSAGE + "\n" + DEVELOPER_MESSAGE_1 + f"\n<|start|>user<|message|>Premise: {p}\nHypothesis: {h}\nOutput:<|end|>"
        #print("Generated prompt:")
        #print(prompt)
        prompts.append(prompt)
        
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=2000,
        #stop=["\n", "</s>"],
    )

    # vLLM: generate returns an iterator; you used earlier llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params)

    predictions_normalized = []
    # collect and normalize
    for i, out in enumerate(outputs):
        # --- Robust final answer detection ---
        text = out.outputs[0].text.strip()
        m = re.findall(
            r"(?:assistant\s*final|assistantfinal)\s*([ECN])\b",
            text,
            flags=re.IGNORECASE
        )

        if m:
            pred = m[-1].upper()
        else:
            # 2. Fallback: check if the entire output ends with a valid label
            m2 = re.findall(r"\b([ECN])\b\s*$", text, flags=re.IGNORECASE)
            pred = m2[-1].upper() if m2 else "UNKNOWN_ERROR"

        predictions_normalized.append(pred)
        rows_for_csv.append((p, h, None, refs[i]))
        # update CSV row
        p, h, _, ref = rows_for_csv[i]
        rows_for_csv[i] = (p, h, pred, ref)

    # write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Premise", "Hypothesis", "Predictions", "References"])
        writer.writerows(rows_for_csv)

    return predictions_normalized, refs

# ================================
# ENTRYPOINT
# ================================
def main():
    # load tokenizer (not strictly necessary for vLLM but kept)
    print("Loading tokenizer (for compatibility)...")
    _ = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading vLLM model...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        #dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=8192,
    )

    # load dataset (json list)
    print(f"Loading dataset: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of samples.")

    print("Dataset size:", len(data))

    # run inference
    print("Running inference...")
    preds, refs = generate_predictions_vllm(llm, data)

    # sample outputs
    print("\n=== SAMPLE OUTPUTS (first 10) ===")
    for i in range(min(10, len(preds))):
        print("REF:", refs[i], " PRED:", preds[i])
    print("-" * 40)

    # metrics
    print("\n📊 Evaluation Results:")
    print("Accuracy :", accuracy_score(refs, preds))
    print("F1 Score :", f1_score(refs, preds, average="weighted", zero_division=0))
    print("Recall   :", recall_score(refs, preds, average="weighted", zero_division=0))
    print("Precision:", precision_score(refs, preds, average="weighted", zero_division=0))

    simple_acc = float(np.mean(np.array(preds) == np.array(refs)))
    print("Accuracy (simple np.mean)  :", simple_acc)

    print("\n🔢 Reference label distribution:")
    print(pd.Series(refs).value_counts().to_string())

    print("\n🔢 Prediction label distribution:")
    print(pd.Series(preds).value_counts().to_string())

    # SBERT similarity (optional)
    if SBERT_AVAILABLE:
        # use textual labels for similarity (not ideal; optional)
        try:
            sim = compute_sbert_similarity(refs, preds)
            print(f"\n🔵 SBERT Similarity Mean: {sim:.4f}")
        except Exception as e:
            print("SBERT similarity error:", e)
    else:
        print("\n🔵 SBERT not available (sentence-transformers not installed).")

if __name__ == "__main__":
    main()
