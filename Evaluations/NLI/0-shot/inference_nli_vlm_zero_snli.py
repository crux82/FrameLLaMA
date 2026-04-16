
import os
import csv
import json
from typing import List, Tuple
import numpy as np
import pandas as pd

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
#MODEL_PATH = "merged_finetuned_llama3.1_8b_fp16_vllm_full" 
#MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"   # change as needed
#MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#MODEL_PATH = "google/gemma-3-4b-it"
MODEL_PATH = "openai/gpt-oss-20b"
TEST_FILE = "Evaluation/NLI/snli_matched_FrameNet.json"               # change as needed (json list)
OUTPUT_CSV = "Evaluation/Pred/NLI/adapted_snli_0shot_1.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# labels expected in reference file (SNLI style)
REF_LABELS = {"entailment", "contradiction", "neutral"}

# ================================
# HELPERS
# ================================
def normalize_reference_label(x: str) -> str:
    """Normalize reference labels to one of 'entailment','contradiction','neutral'."""
    if x is None:
        return "unknown"
    t = str(x).strip().lower()
    if t.startswith("ent"):
        return "entailment"
    if t.startswith("contr"):
        return "contradiction"
    if t.startswith("neu"):
        return "neutral"
    # fallback if literal E/C/N
    if t == "e":
        return "entailment"
    if t == "c":
        return "contradiction"
    if t == "n":
        return "neutral"
    # final fallback
    return "neutral"

def normalize_model_output(text: str) -> str:
    """
    Normalize arbitrary model output text to one of SNLI labels.
    Handles outputs like:
      "E", "Entailment", "Output: E", "Answer: entailment", "Entailment.\nExplanation: ..."
    """
    if text is None:
        return "neutral"
    t = str(text).strip().lower()

    # remove common noise tokens
    noise = ["</s>", "</s", "<s>", "output:", "answer:", "prediction:", "pred:", "label:"]
    for n in noise:
        t = t.replace(n, " ")

    # remove punctuation at ends
    t = t.strip(" .\n\r\t:;")

    # direct checks
    if t == "":
        return "neutral"
    # exact tokens
    if t in {"e", "entailment", "entails"}:
        return "entailment"
    if t in {"c", "contradiction", "contradictory"}:
        return "contradiction"
    if t in {"n", "neutral", "neither"}:
        return "neutral"

    # look for keywords anywhere (safe order: check longer words first)
    if "entail" in t:
        return "entailment"
    if "contrad" in t:
        return "contradiction"
    if "neutral" in t or "may" in t or "could" in t:
        return "neutral"

    # fallback by first letter
    first = t[0]
    if first == "e":
        return "entailment"
    if first == "c":
        return "contradiction"
    if first == "n":
        return "neutral"

    # final safe default
    return "neutral"

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
PROMPT_TEMPLATE = """# Task: Judge if the hypothesis necessarily follows from the premise.
        # Consider the truth value of the premise. If the premise is true, does it necessarily mean that the hypothesis must also be true?

        # Output:
        E — choose this option if the hypothesis is necessarily true in every case.
        C — choose this option if the hypothesis is always untrue and cannot be correct.
        N — choose this option if the hypothesis can be true in some situations and false in others.
        # Do not output anything other than letters N, C, or E.

        Premise: {premise}
        Hypothesis: {hypothesis}
        # Output:"""

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
        prom = PROMPT_TEMPLATE.format(premise=p, hypothesis=h)
        #promp = prom + "\n" + prom + "\n" + prom + "\n" + prom  # repeat 4 times to reduce randomness
        #print("Generated prompt:")
        #print(promp)
        prompts.append(prom)
        refs.append(normalize_reference_label(item.get("gold_label", "")))
        #refs.append(item.get("gold_label", ""))
        rows_for_csv.append((p, h, None, normalize_reference_label(item.get("gold_label", ""))))

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=2,
        stop=["\n", "</s>"],
    )

    # vLLM: generate returns an iterator; you used earlier llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params)

    predictions_normalized = []
    # collect and normalize
    for i, out in enumerate(outputs):
        # out.outputs is a list of Output objects; join their text if multiple
        # Usually out.outputs[0].text is sufficient
        try:
            raw_text = out.outputs[0].text
            #print(raw_text)
        except Exception:
            # fallback: try str(out)
            raw_text = str(out)
        norm_pred = normalize_model_output(raw_text)
        predictions_normalized.append(norm_pred)
        # update CSV row
        p, h, _, ref = rows_for_csv[i]
        rows_for_csv[i] = (p, h, norm_pred, ref)

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
