"""Tests for the evaluation helpers and repository layout (no GPU needed)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama import nli, srl  # noqa: E402
from framellama.config import load_config, repo_path  # noqa: E402


def test_config_evaluation_files_exist():
    ev = load_config(ROOT / "configs" / "acl2026.yaml")["evaluation"]
    for path in ev["nli"]["benchmarks"].values():
        assert repo_path(path).exists(), path
    assert repo_path(ev["srl"]["gold_file"]).exists()


def test_prompt_files_have_placeholders():
    for (setting, bench), name in nli.PROMPT_FILES.items():
        text = nli.read_prompt(name)
        assert "{premise}" in text and "{hypothesis}" in text, name


def test_nli_files_load_and_labels_normalise():
    confer = nli.load_nli_file(ROOT / "evaluation/splits/nli/test_confer.json")
    snli = nli.load_nli_file(ROOT / "evaluation/splits/nli/test_snli.json")
    assert len(confer) == 2348 and len(snli) == 9824
    assert {nli.gold_letter(x["gold_label"]) for x in confer} == {"E", "C", "N"}
    assert {nli.gold_letter(x["gold_label"]) for x in snli} == {"E", "C", "N"}
    assert all(x["premise"] and x["hypothesis"] for x in snli[:100])


def test_nli_prompts_and_scoring():
    items = [{"premise": "A man sleeps.", "hypothesis": "A man is awake.", "gold_label": "contradiction"}]
    for setting in ("zero-shot", "few-shot"):
        prompt = nli.build_prompts(items, setting, "snli")[0]
        assert "A man sleeps." in prompt and "{premise}" not in prompt
    preds, refs = nli.score_outputs([" C"], items, "zero-shot")
    assert preds == ["contradiction"] and refs == ["contradiction"]
    preds, refs = nli.score_outputs([" C"], items, "few-shot")
    assert preds == ["C"] and refs == ["C"]
    assert nli.extract_gptoss_label("analysis...assistantfinalN") == "N"


def test_normalize_model_output_cases():
    assert nli.normalize_model_output("E") == "entailment"
    assert nli.normalize_model_output("Output: C") == "contradiction"
    assert nli.normalize_model_output("") == "neutral"


def test_srl_inputs_cover_every_gold_annotation():
    gold = srl.load_gold(ROOT / "evaluation/splits/srl/srl_gold_std.json")
    inputs = srl.prepare_inputs(gold)
    assert len(inputs) == sum(len(s["annotations"]) for s in gold) == 5694
    for task in srl.TASKS:
        assert inputs[0]["sentence"] in srl.build_prompt(task, inputs[0])


def test_srl_align_keeps_parsed_answers_only():
    gold = [{"input_sentence": "S1", "annotations": [
        {"frame": "Food", "lexical_unit": "food", "frame_elements": {"Food": "food"}},
        {"frame": "Goal", "lexical_unit": "into", "frame_elements": {"Landmark": "Iraq"}}]}]
    inputs = srl.prepare_inputs(gold)
    good = srl.parse_output("roles", '{"input_sentence": "S1", "annotations": [{"frame": "Food", '
                                     '"lexical_unit": "food", "frame_elements": {"Food": ["food"]}}]}', inputs[0])
    bad = srl.parse_output("roles", "no json here", inputs[1])
    result, gold_std = srl.align("roles", inputs, [good, bad], gold)
    assert result[0]["annotations"][0]["frame_elements"] == {"Food": "food"}
    assert gold_std[0]["annotations"] == [gold[0]["annotations"][0]]
    m = srl.score("roles", result, gold_std)
    assert abs(m["roles_f1"] - 1.0) < 1e-9


def test_released_srl_results_are_valid_json():
    for path in (ROOT / "results/acl2026/srl").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list) and data and "annotations" in data[0], path.name


def test_data_sources_default_to_hugging_face():
    from framellama.config import data_source
    cfg = load_config(ROOT / "configs" / "acl2026.yaml")
    repo = "hf://datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions/"
    assert cfg["training"]["train_data"] == repo + "train.json"
    assert cfg["evaluation"]["qa"]["test_files"] == {
        "opened": repo + "test/Opened.json", "closed": repo + "test/Closed.json", "mcq": repo + "test/MCQ.json"}
    assert "eval_file" not in cfg["training"]
    for spec in (repo + "train.json",
                 "https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions/resolve/main/train.json"):
        assert data_source(spec, from_cli=True) == spec


def test_example_data_files_have_released_fields():
    from framellama.config import load_json_data
    tr = load_config(ROOT / "configs" / "acl2026.yaml")["training"]
    for name in ("train.json", "test/Opened.json"):
        ds = load_json_data(str(ROOT / "data" / "train" / name))
        assert len(ds) > 0
        for rec in ds:
            for field in (tr["instruction_field"], tr["question_field"], tr["answer_field"], "Task Family"):
                assert isinstance(rec.get(field), str) and rec[field].strip(), (name, field)


def test_answer_type_rule():
    from framellama.qa import answer_type
    assert answer_type("Yes") == "closed" and answer_type(" No ") == "closed"
    assert answer_type("c) Buyer defined as: ...") == "mcq"
    assert answer_type("The \"Killing\" frame represents ...") == "opened"


def test_released_predictions_can_be_scored():
    import subprocess
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout  # noqa: E731
    out = run("scripts/evaluate_nli.py", "--score-file", "results/acl2026/nli/adapted_nli_0shot_confer.csv")
    assert "Accuracy : 0.55" in out
    out = run("scripts/evaluate_srl.py", "--score-only", "--task", "roles",
              "--result-file", "results/acl2026/srl/baseline_srl_result_roles.json")
    assert '"roles_f1": 0.17' in out
    out = run("scripts/evaluate_qa.py", "--score-file", "results/acl2026/qa/baseline_closed.csv")
    assert "accuracy 0.6" in out


def test_adapted_and_baseline_flags():
    import argparse
    from framellama.config import add_model_args, select_models
    ev = load_config(ROOT / "configs" / "acl2026.yaml")["evaluation"]
    p = argparse.ArgumentParser()
    add_model_args(p)
    assert select_models(p.parse_args(["--adapted"]), ev) == [
        ("adapted", "sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17")]
    assert select_models(p.parse_args(["--baseline"]), ev) == [
        ("baseline", "meta-llama/Llama-3.1-8B-Instruct")]
    assert [k for k, _ in select_models(p.parse_args([]), ev)] == ["baseline", "adapted"]
    assert select_models(p.parse_args(["--models", "openai/gpt-oss-20b"]), ev) == [
        ("openai_gpt-oss-20b", "openai/gpt-oss-20b")]
