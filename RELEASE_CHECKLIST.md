# Release checklist (delete this file before publishing)

## 1. Upload the training data to Hugging Face

Dataset repository: https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions

```bash
huggingface-cli login
huggingface-cli upload sag-uniroma2/FrameLLaMA-FullFN17-Instructions train.json train.json --repo-type dataset
huggingface-cli upload sag-uniroma2/FrameLLaMA-FullFN17-Instructions test.json  test.json  --repo-type dataset
```

- [ ] Check the files have the fields `Instruction`, `Question`, `Answer`, `Task Family`:
      `python tools/dataset_stats.py train.json test.json` (also reports missing or empty fields).
- [ ] Fill the two `TO_FILL` tables in `docs/huggingface/dataset_card.md` with the tables printed by the same command,
      and the counts / SHA-256 values in `data/manifests/acl2026.json`.
- [ ] Upload the card: `huggingface-cli upload sag-uniroma2/FrameLLaMA-FullFN17-Instructions docs/huggingface/dataset_card.md README.md --repo-type dataset`
- [ ] Test loading: `python -c "from datasets import load_dataset; print(load_dataset('sag-uniroma2/FrameLLaMA-FullFN17-Instructions'))"`
- [ ] Put the dataset's commit hash in `data/manifests/acl2026.json` (`dataset_revision`).

- [ ] Check that `dataset_stats.py` reports about 59.4k open-ended, 81.7k yes/no and 106.8k multiple-choice questions in `test.json` (the paper's numbers).

- [ ] Check that training starts from Hugging Face on the GPU server:
      `python scripts/train.py --config configs/acl2026.yaml --max-steps 5`
      (the log line "Loading training data: hf://datasets/..." should be followed by the number of examples).

## 2. Model card

- [ ] Paste `docs/huggingface/model_card.md` into the model page:
      `huggingface-cli upload sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17 docs/huggingface/model_card.md README.md`
- [ ] It says the checkpoint holds merged weights. Check the model's "Files and versions" tab:
      `model-0000x-of-0000y.safetensors` = merged (correct), `adapter_model.safetensors` = adapter (change the "Checkpoint" row).

## 3. Files from the server

- [ ] **Four corrupted SRL files.** `adapted_srl_result_LUs.json`, `adapted_srl_result_roles.json`,
      `baseline_srl_gold-std_roles.json`, `baseline_srl_result_LUs.json` contain only zero bytes in the uploaded repository.
      Copy intact versions into `results/acl2026/srl/`.
- [ ] **Training package versions.** `scripts/train.py` needs `trl.DataCollatorForCompletionOnlyLM` (removed in TRL 0.20);
      `environment.yml` has TRL 0.24.0. Run `pip freeze` in the training environment and put the versions of unsloth, trl,
      peft, transformers, bitsandbytes and torch in `requirements-train.txt`.
- [ ] **Optional:** if you still have the script that assembled the per-frame CSVs into `train.json` / `test.json`
      (it adds `Instruction` and `Task Family`), add it to `tools/`; the documentation then no longer needs to describe that step in words.

## 4. Run once on the GPU server

- [ ] `python examples/nli_demo.py`
- [ ] `python scripts/evaluate_nli.py --models framellama --benchmarks confer --settings zero-shot` (paper: 0.55)
- [ ] `python scripts/evaluate_srl.py --models framellama --tasks roles --limit 20`
- [ ] `python scripts/evaluate_qa.py --models framellama --task closed --limit 100` (reads test.json from Hugging Face)
- [ ] Data sample: run the two commands under "Sample records" in `data/README.md`.

## 5. Before making everything public

- [ ] Licences: confirm with Danilo/Roberto that the FrameNet-derived corpus, CONFER, the SNLI subset and the SRL gold data
      may be redistributed; complete the licence lines in the dataset card and `evaluation/README.md`.
- [ ] SNLI subset: add one sentence to `evaluation/README.md` saying how `test_snli.json` (9,824 pairs) was selected.
- [ ] Remove the "ACTION FOR SHAHID" boxes (README.md, evaluation/README.md, results/acl2026/README.md) and delete this file.
