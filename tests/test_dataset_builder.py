"""Tests for the dataset generator.

Fast tests need no FrameNet and no GPU. The end-to-end test generates one
frame and runs only when FRAMELLAMA_SLOW_TESTS=1 (it loads all of FrameNet and
the sentence-embedding model, which takes a few minutes).
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framellama import dataset_builder, templates  # noqa: E402

RECORD_FIELDS = {
    "Question": str, "Answer": str, "Task Family": str, "task_id": str, "task_key": str,
    "frame_name": str, "question_id": str, "question_template": str, "answer_type": str,
    "answer_template": str, "source": str,
}


def _load_cli():
    spec = importlib.util.spec_from_file_location("generate_dataset", ROOT / "scripts" / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task_keys_in_code():
    src = (ROOT / "src" / "framellama" / "dataset_builder.py").read_text(encoding="utf-8")
    return re.findall(r'csv_data\.append\(\[\s*[^,\n]+,\s*"([^"]+)",\s*"([^"]+)"', src)


def check_record(rec):
    for field, typ in RECORD_FIELDS.items():
        assert field in rec, f"missing field {field}"
        assert isinstance(rec[field], typ), f"{field} should be {typ.__name__}"
    assert rec["Question"].strip() and rec["Answer"].strip()
    assert rec["task_key"] == f"{rec['Task Family']} | Q{rec['question_id']}"
    assert rec["task_id"] in {f"T{i}" for i in range(1, 41)}
    assert rec["source"] == "FrameNet 1.7"


def test_forty_task_families_in_code():
    keys = _task_keys_in_code()
    assert len(set(keys)) == 40, "the paper describes 40 task families"
    assert set(templates.PARAPHRASED_TASKS) <= set(keys)


def test_task_family_table_matches_code():
    from framellama.task_families import TASK_FAMILIES
    keys = {f"{p} | Q{q}" for p, q in _task_keys_in_code()}
    assert set(TASK_FAMILIES) == keys
    numbers = [v for v in TASK_FAMILIES.values() if v]
    assert len(numbers) == len(set(numbers)), "duplicate task numbers"


def test_template_sections_have_text():
    assert isinstance(templates.TEMPLATE, dict) and len(templates.TEMPLATE) > 20
    blob = json.dumps(templates.TEMPLATE)
    assert "{frame_name}" in blob


def test_merge_passes_keeps_second_pass_only_for_paraphrased_tasks():
    row = lambda pt, q, text: ["F", pt, q, "id", "Opened", "id", text, "a"]  # noqa: E731
    first = [row("Frame Definition", "1", "p1"), row("Lexical Units", "1", "p1")]
    second = [row("Frame Definition", "1", "p2"), row("Lexical Units", "1", "p2")]
    merged = dataset_builder.merge_passes(first, second)
    texts = [(r[1], r[6]) for r in merged]
    assert texts == [("Frame Definition", "p1"), ("Frame Definition", "p2"), ("Lexical Units", "p1")]


def test_pick_follows_variant():
    dataset_builder.VARIANT = 0
    assert dataset_builder.pick(["a", "b"]) == "a"
    dataset_builder.VARIANT = 1
    assert dataset_builder.pick(["a", "b"]) == "b"
    dataset_builder.VARIANT = 0


def test_row_to_record_schema():
    cli = _load_cli()
    row = ["Commerce sell", "Frame Definition", "1", "FDQ1", "Opened", "FDQ1", "Q?", "A."]
    rec = cli.row_to_record(row, {"Frame Definition | Q1": "T1"})
    check_record(rec)
    assert rec["task_id"] == "T1" and rec["Task Family"] == "Frame Definition"


def test_missing_framenet_dir_gives_clear_error(tmp_path):
    from framellama import framenet_reader
    with pytest.raises(FileNotFoundError, match="frameIndex.xml"):
        framenet_reader.validate_framenet_dir(str(tmp_path))
    (tmp_path / "frameIndex.xml").write_text("<x/>")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        framenet_reader.validate_framenet_dir(str(tmp_path))


def _framenet_available():
    try:
        import nltk
        import sentence_transformers  # noqa: F401
        nltk.data.find("corpora/framenet_v17")
        nltk.data.find("corpora/wordnet")
        return True
    except Exception:
        return False


@pytest.mark.skipif(os.environ.get("FRAMELLAMA_SLOW_TESTS") != "1", reason="set FRAMELLAMA_SLOW_TESTS=1")
@pytest.mark.skipif(not _framenet_available(), reason="needs NLTK framenet_v17, wordnet, sentence-transformers")
def test_generate_one_frame_end_to_end(tmp_path):
    import csv as csvmod
    out = tmp_path / "one.jsonl"
    cmd = [sys.executable, str(ROOT / "scripts" / "generate_dataset.py"),
           "--frames", "Abandonment", "--output-dir", str(tmp_path), "--jsonl", str(out)]
    subprocess.run(cmd, check=True, cwd=tmp_path)
    with open(tmp_path / "Abandonment.csv", encoding="utf-8") as f:
        rows = list(csvmod.reader(f))
    assert rows[0] == dataset_builder.CSV_COLUMNS and len(rows) > 1
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(rows) - 1
    for rec in records:
        check_record(rec)
        assert rec["frame_name"] == "Abandonment"
    summary = json.loads((tmp_path / "_summary.json").read_text())
    assert summary["num_examples"] == len(records)
    assert not summary["failed_frames"]
