"""NLI prompts, label normalisation and metrics.

The prompt texts live in evaluation/prompts/ and were extracted verbatim from
the original evaluation scripts. The label-normalisation functions below are
copied from those scripts unchanged.

Settings (as in the original scripts)
-------------------------------------
zero-shot : raw prompt, greedy, max_tokens=2, stop at newline. Model output
            and gold label are normalised to entailment/contradiction/neutral
            with normalize_model_output / normalize_reference_label.
few-shot  : raw prompt with 7 fixed CONFER exemplars, greedy, max_tokens=1.
            The generated letter is compared with the gold letter (E/C/N).
gpt-oss   : harmony-format prompt for openai/gpt-oss-20b baselines, up to 2000
            tokens; the final E/C/N is read from the "assistantfinal" channel.
"""

import json
import re
from datetime import date
from pathlib import Path

from .config import REPO_ROOT

PROMPT_DIR = REPO_ROOT / "evaluation" / "prompts"

PROMPT_FILES = {
    ("zero-shot", "confer"): "nli_zero_shot_confer.txt",
    ("zero-shot", "snli"): "nli_zero_shot_snli.txt",
    ("few-shot", "confer"): "nli_few_shot_confer.txt",
    ("few-shot", "snli"): "nli_few_shot_snli.txt",
}
GPTOSS_DEVELOPER = {
    "zero-shot": "nli_zero_shot_gptoss_developer.txt",
    "few-shot": "nli_few_shot_gptoss_developer.txt",
}
GPTOSS_SYSTEM = "nli_gptoss_system.txt"

LETTER = {"entailment": "E", "contradiction": "C", "neutral": "N"}


def read_prompt(name):
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def fill(template, premise, hypothesis):
    return template.replace("{premise}", premise).replace("{hypothesis}", hypothesis)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_nli_file(path):
    """Load a JSON list of NLI items as dicts with premise, hypothesis, gold_label.

    Accepts the CONFER format (premise/hypothesis, gold E/C/N) and the SNLI
    format (sentence1/sentence2, gold entailment/contradiction/neutral).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of examples.")
    items = []
    for i, ex in enumerate(data):
        premise = ex.get("premise", ex.get("sentence1"))
        hypothesis = ex.get("hypothesis", ex.get("sentence2"))
        if premise is None or hypothesis is None:
            raise ValueError(f"{path}: example {i} has no premise/hypothesis (or sentence1/sentence2).")
        items.append({
            "premise": str(premise).strip(),
            "hypothesis": str(hypothesis).strip(),
            "gold_label": ex.get("gold_label", ""),
        })
    return items


# ---------------------------------------------------------------
# Label normalisation (verbatim from the original scripts)
# ---------------------------------------------------------------
def normalize_reference_label(x: str) -> str:
    """Normalize reference labels to one of 'entailment','contradiction','neutral'."""
    if x is None:
        return "neutral"
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


def gold_letter(x):
    """Gold label as E/C/N (identity for CONFER, converts SNLI word labels)."""
    return LETTER[normalize_reference_label(x)]


def extract_gptoss_label(text):
    """Final E/C/N from a gpt-oss answer (from the original gpt-oss scripts)."""
    text = text.strip()
    m = re.findall(r"(?:assistant\s*final|assistantfinal)\s*([ECN])\b", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m2 = re.findall(r"\b([ECN])\b\s*$", text, flags=re.IGNORECASE)
    return m2[-1].upper() if m2 else "UNKNOWN_ERROR"


# ---------------------------------------------------------------
# Prompting and decoding
# ---------------------------------------------------------------
def build_prompts(items, setting, benchmark, prompt_format="raw"):
    if prompt_format == "gpt-oss":
        system = read_prompt(GPTOSS_SYSTEM).replace("{current_date}", date.today().isoformat())
        developer = read_prompt(GPTOSS_DEVELOPER[setting])
        return [system + "\n" + developer +
                f"\n<|start|>user<|message|>Premise: {it['premise']}\nHypothesis: {it['hypothesis']}\nOutput:<|end|>"
                for it in items]
    template = read_prompt(PROMPT_FILES[(setting, benchmark)])
    return [fill(template, it["premise"], it["hypothesis"]) for it in items]


def sampling_kwargs(setting, prompt_format="raw"):
    if prompt_format == "gpt-oss":
        return dict(temperature=0.0, top_p=0.9, max_tokens=2000)
    if setting == "zero-shot":
        return dict(temperature=0.0, top_p=0.9, max_tokens=2, stop=["\n", "</s>"])
    return dict(temperature=0.0, top_p=0.9, max_tokens=1, stop=["</s>"])


def score_outputs(raw_outputs, items, setting, prompt_format="raw"):
    """Return (predictions, references) in the label space used by the setting."""
    if prompt_format == "gpt-oss":
        preds = [extract_gptoss_label(t) for t in raw_outputs]
        refs = [gold_letter(it["gold_label"]) for it in items]
    elif setting == "zero-shot":
        preds = [normalize_model_output(t) for t in raw_outputs]
        refs = [normalize_reference_label(it["gold_label"]) for it in items]
    else:
        preds = [t.strip() for t in raw_outputs]
        refs = [gold_letter(it["gold_label"]) for it in items]
    return preds, refs


def metrics(preds, refs):
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
    from collections import Counter
    return {
        "n": len(refs),
        "accuracy": accuracy_score(refs, preds),
        "f1_weighted": f1_score(refs, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(refs, preds, average="weighted", zero_division=0),
        "precision_weighted": precision_score(refs, preds, average="weighted", zero_division=0),
        "per_class": classification_report(refs, preds, output_dict=True, zero_division=0),
        "reference_distribution": dict(Counter(refs)),
        "prediction_distribution": dict(Counter(preds)),
    }
