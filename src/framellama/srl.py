"""SRL evaluation pipeline: inputs -> prompts -> parsed outputs -> aligned files.

Prompts and parsers come verbatim from the original scripts (srl_prompts.py);
scoring code comes verbatim from the original F1 scripts
(srl_scoring_roles.py, srl_scoring_lus.py).

Two steps were not in the original repository and are reconstructed here
from the released files:

* prepare_inputs(): one model query per gold annotation of
  srl_gold_std.json, with the other frames / lexical units of the sentence as
  context (the fields read by the original scripts: sentence, frame,
  lexical_unit, frame_combine, lexical_combine).
* align(): groups the parsed answers back into sentences and keeps, for each
  sentence, only the gold annotations whose query produced a parseable
  answer. This matches the released *_gold-std_*.json files, which are
  order-preserving subsets of srl_gold_std.json.
"""

import json

from . import srl_prompts as P

TASKS = ("roles", "lu", "frame")


def load_gold(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prepare_inputs(gold):
    """One query per gold annotation."""
    inputs = []
    for si, sent in enumerate(gold):
        anns = sent.get("annotations", [])
        frames = [a.get("frame", "") for a in anns]
        lus = [a.get("lexical_unit", "") for a in anns]
        for ai, ann in enumerate(anns):
            inputs.append({
                "sentence_index": si,
                "annotation_index": ai,
                "sentence": sent["input_sentence"],
                "frame": ann.get("frame", ""),
                "lexical_unit": ann.get("lexical_unit", ""),
                "frame_combine": frames,
                "lexical_combine": lus,
            })
    return inputs


def build_prompt(task, sample):
    """Same prompt construction as the original generate_predictions_vllm()."""
    frame = sample["frame"].replace("_", " ")
    lexical_unit = sample["lexical_unit"]
    sentence = sample["sentence"]
    if task == "roles":
        frame_combine = ", ".join(sample["frame_combine"])
        lexical_combine = ", ".join(sample["lexical_combine"])
        return P.build_prompt_roles(frame_combine, lexical_combine, frame, sentence, lexical_unit)
    if task == "lu":
        return P.build_prompt_lu(frame, sentence, lexical_unit)
    if task == "frame":
        return P.build_prompt_frame(frame, sentence, lexical_unit)
    raise ValueError(task)


def sampling_kwargs():
    return dict(temperature=0.0, top_p=1.0, max_tokens=512, stop=["```", "</s>"])


def parse_output(task, raw_text, sample):
    """Same parsing as the original scripts. Returns a dict/list, or the raw text on failure."""
    frame = sample["frame"].replace("_", " ")
    if task == "roles":
        try:
            return P.safe_parse_json(raw_text)
        except Exception:
            return raw_text
    raw_text = raw_text.strip()
    json_blocks = P.extract_json_blocks(raw_text)
    cleaned_blocks = P.remove_newlines_from_output(json_blocks)
    filtered = P.filter_json_blocks_by_frame(cleaned_blocks, frame)
    return filtered if filtered else raw_text


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value)
    return json.dumps(value, ensure_ascii=False)


def predicted_annotation(task, parsed, sample):
    """The annotation predicted for one query, or None if the answer could not be parsed."""
    if isinstance(parsed, list):          # lu / frame: list of parsed JSON objects
        parsed = parsed[0] if parsed else None
    if not isinstance(parsed, dict):
        return None
    anns = parsed.get("annotations")
    if not isinstance(anns, list) or not anns or not isinstance(anns[0], dict):
        return None
    target = sample["frame"].replace("_", " ")
    ann = next((a for a in anns if isinstance(a, dict) and a.get("frame") == target), anns[0])
    ann = dict(ann)
    if isinstance(ann.get("frame_elements"), dict):
        # the scorer calls .strip() on every span; make sure spans are strings
        ann["frame_elements"] = {str(k): _as_text(v) for k, v in ann["frame_elements"].items()}
    elif "frame_elements" in ann:
        ann["frame_elements"] = {}
    for key in ("frame", "lexical_unit"):
        if key in ann and not isinstance(ann[key], str):
            ann[key] = _as_text(ann[key])
    return ann


def align(task, inputs, parsed_outputs, gold):
    """Group answers by sentence. Returns (result_file, gold_std_file) lists."""
    by_sentence = {}
    for sample, parsed in zip(inputs, parsed_outputs):
        ann = predicted_annotation(task, parsed, sample)
        if ann is None:
            continue
        si, ai = sample["sentence_index"], sample["annotation_index"]
        by_sentence.setdefault(si, []).append((ai, ann))

    result, gold_std = [], []
    for si in sorted(by_sentence):
        pairs = sorted(by_sentence[si], key=lambda x: x[0])
        sentence = gold[si]["input_sentence"]
        result.append({"input_sentence": sentence, "annotations": [a for _, a in pairs]})
        gold_std.append({"input_sentence": sentence,
                         "annotations": [gold[si]["annotations"][ai] for ai, _ in pairs]})
    return result, gold_std


def score(task, result, gold_std):
    """Run the original scoring code (see srl_scoring_*.py for the argument order)."""
    if task == "roles":
        from .srl_scoring_roles import score_roles
        return score_roles(result, gold_std)
    from .srl_scoring_lus import score_lus
    return score_lus(result, gold_std)


def align_to_gold(result, gold):
    """Build the aligned gold file for an existing result file.

    Each predicted annotation is matched to an unused gold annotation of the
    same sentence with the same frame and lexical unit (case-insensitive).
    Unmatched predictions and sentences not in the gold file are left out.
    Returns (result, gold_std) lists aligned sentence by sentence.
    """
    by_sentence = {}
    for sent in gold:
        by_sentence.setdefault(sent["input_sentence"], sent)
    res_out, gold_out = [], []
    for sent in result:
        g = by_sentence.get(sent.get("input_sentence"))
        if g is None:
            continue
        used, pred_keep, gold_keep = set(), [], []
        for ann in sent.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            for j, gann in enumerate(g["annotations"]):
                if j in used:
                    continue
                if (str(gann.get("frame", "")).lower() == str(ann.get("frame", "")).lower()
                        and str(gann.get("lexical_unit", "")).lower() == str(ann.get("lexical_unit", "")).lower()):
                    used.add(j)
                    pred_keep.append(ann)
                    gold_keep.append(gann)
                    break
        res_out.append({"input_sentence": sent["input_sentence"], "annotations": pred_keep})
        gold_out.append({"input_sentence": sent["input_sentence"], "annotations": gold_keep})
    return res_out, gold_out
