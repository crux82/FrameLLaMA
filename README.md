# FrameLLaMA: Frame-Semantic Knowledge Injection for Event-Level Inference in LLMs

**Official code, model, and dataset repository for our ACL 2026 paper:**

**[Frame-Semantic Knowledge Injection for Event-Level Inference in LLMs](https://aclanthology.org/2026.acl-short.55/)**  
*Shahid Iqbal Rai, Danilo Croce, Roberto Basili — ACL 2026 (Short Papers)*

[Paper](https://aclanthology.org/2026.acl-short.55/) · [Fine-tuned Model](https://huggingface.co/sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17) · [Dataset](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions) · [Code](https://github.com/crux82/FrameLLaMA)


---

## 1. Overview

Large Language Models (LLMs) exhibit remarkable language understanding capabilities, but they can still struggle with reasoning about the structure of events and the semantic relationships between their participants.

For example, understanding that *Alice sold a bicycle to Bob* entails that *Bob bought a bicycle from Alice* requires recognizing the relationship between two descriptions of the same transaction.

More generally, semantic reasoning may depend on:

- Identifying which event or situation is described by a sentence.
- Recognizing the semantic roles played by its participants.
- Understanding constraints on the entities that can fill these roles.
- Recognizing relations between events, including inheritance, causation, and perspective.
- Distinguishing semantically compatible situations from incompatible ones.

**Frame Semantics**, as represented in [FrameNet](https://framenet.icsi.berkeley.edu/), provides structured knowledge about these phenomena through frames, frame elements, lexical units, and frame-to-frame relations.

### Our approach

FrameLLaMA investigates whether this structured knowledge can be transferred into an LLM through parameter-efficient instruction tuning.

Rather than fine-tuning directly on downstream Natural Language Inference (NLI) or Semantic Role Labeling (SRL) datasets, we construct instruction-style supervision from FrameNet 1.7. The supervision moves from fact-oriented prompting to *principle-oriented* supervision over the full FrameNet inventory: it encodes semantic constraints through semantic types, sense-aware definitions, frame relations, and role-annotated examples, together with carefully filtered negative examples.

The methodology consists of four steps:

1. **Knowledge extraction:** extract frames, semantic roles, lexical information, semantic types, and frame relations from FrameNet 1.7.
2. **Knowledge textification:** convert structured semantic information into natural-language instruction–response examples through 40 task families.
3. **Model adaptation:** fine-tune Meta-Llama-3.1-8B-Instruct using Low-Rank Adaptation (LoRA) on more than 491,000 generated examples.
4. **Evaluation:** use NLI (CONFER and an SNLI diagnostic subset) as a diagnostic of event-level reasoning, complemented by SRL analyses of frame, role and span structure.

The experiments show improvements in several event-semantic evaluations, particularly for entailment and contradiction judgments and for aspects of semantic role prediction.

The goal is not to construct a universally stronger LLM, but to investigate how structured semantic supervision affects event-level semantic competence.

---

## 2. Quick Start: Run an NLI Example

### 2.1 Installation

Requirements: Linux, Python 3.10, and an NVIDIA GPU with about 18 GB of free memory (the 8B model is loaded in 16-bit with [vLLM](https://github.com/vllm-project/vllm)).

```bash
git clone https://github.com/crux82/FrameLLaMA.git
cd FrameLLaMA

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, recreate the exact environment used for the paper's evaluation with conda:

```bash
conda env create -f environment.yml
conda activate framellama
```

To also fine-tune the model (Section 6.1), install the training packages into the same environment:

```bash
pip install -r requirements-train.txt
```

> **This setup is done once.** All commands in the rest of this README (demo, dataset generation, training, evaluation) assume you are in the repository folder with this environment activated (`source .venv/bin/activate` or `conda activate framellama`); nothing needs to be installed again.

The base model `meta-llama/Llama-3.1-8B-Instruct` is gated on Hugging Face. To run the base-model comparisons, accept its license and log in with `huggingface-cli login`.

### 2.2 Example: Event-Level Natural Language Inference

**Premise**

> Alice sold a bicycle to Bob.

**Hypothesis**

> Bob bought a bicycle from Alice.

**Expected label**

> Entailment

Although the two sentences use different predicates, they describe compatible perspectives on the same transaction.

Run the example:

```bash
python examples/nli_demo.py
```

The script loads the released checkpoint, builds the prompt used for the zero-shot NLI evaluation in the paper (`evaluation/prompts/nli_zero_shot_confer.txt`), decodes greedily, maps the answer to a label, and prints:

```text
============================================================
FrameLLaMA NLI demonstration
============================================================
Model: sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17

Premise:
Alice sold a bicycle to Bob.

Hypothesis:
Bob bought a bicycle from Alice.

Raw model output: '<model output>'

Predicted label: <model prediction>
Expected label:  Entailment
============================================================
```

Other options:

```bash
# your own example
python examples/nli_demo.py --premise "A woman is playing a violin on a stage." \
    --hypothesis "A woman is playing a musical instrument." --expected entailment

# print the prompt without loading the model (no GPU needed)
python examples/nli_demo.py --dry-run
```

The predicted label is generated by the model; the expected label is shown for reference only. This is a demonstration of model usage, not a reproduction of the evaluation results reported in the paper.

---

## 3. Model and Dataset

The model and training resources are distributed through the [SAG Hugging Face organization](https://huggingface.co/sag-uniroma2).

### 3.1 Fine-Tuned Model

**[FrameLLaMA-3.1-8B-Instruct-FullFN17](https://huggingface.co/sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17)**

| Property | Description |
|---|---|
| Base model | Meta-Llama-3.1-8B-Instruct (4-bit Unsloth build during training) |
| Semantic resource | FrameNet 1.7 |
| Supervision | FrameNet-derived instruction–response examples |
| Task families | 40 |
| Training instances | More than 491,000 |
| Adaptation | LoRA (r = 8, α = 16, dropout = 0.1), Unsloth |
| Prompt format | Alpaca-style instruction prompt (see `scripts/train.py`) |
| Primary evaluation | Natural Language Inference |
| Additional evaluation | Semantic Role Labeling |

The checkpoint can be used directly for inference. The loading code (`src/framellama/vllm_utils.py`) accepts both merged weights and LoRA adapters: for adapters it loads the base model named in `adapter_config.json` with LoRA enabled.

The model card text is kept in `docs/huggingface/model_card.md`.

> **ACTION FOR SHAHID — Model card**
>
> Upload `docs/huggingface/model_card.md` as the model page README (see `RELEASE_CHECKLIST.md`).

### 3.2 Training Dataset

**[sag-uniroma2/FrameLLaMA-FullFN17-Instructions](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions)**

The instruction-tuning corpus used to train the released checkpoint: more than 491,000 English instruction–response pairs generated from all 1,221 frames of FrameNet 1.7 in 40 task families. These are the exact files of the published training run. The dataset card (`docs/huggingface/dataset_card.md`) documents the generation method, fields, splits, provenance and licence.

| Split | File | Instances | Use in the paper |
|---|---|---|---|
| `train` | `train.json` | more than 491K | LoRA fine-tuning |
| `test` | `test/Opened.json` | 59.4k open-ended | in-domain FrameNet QA evaluation (Section 6.4) |
| | `test/Closed.json` | 81.7k yes/no | |
| | `test/MCQ.json` | 106.8k multiple choice | |

The `test` split (247.9k held-out instances in total) is not used in training.

Each file is a JSON list of records with four fields:

| Field | Description |
|---|---|
| `Instruction` | a short generic instruction (e.g. "Please answer the question using correct semantic reasoning."); it does not carry task information |
| `Question` | the FrameNet-derived question: open-ended, Yes/No, or multiple choice |
| `Answer` | the expected response |
| `Task Family` | the kind of FrameNet knowledge the example comes from (e.g. `Frame Definition`, `Semantic Type`, `Uses`) |

```json
{
  "Instruction": "Provide the most appropriate factual response to the question.",
  "Question": "Can you describe what typically happens in the \"Indicating\" frame? What kind of situation or event does the Indicating frame represent?",
  "Answer": "The \"Indicating\" frame represents a type of situation or event in which specific participants (called frame elements) play meaningful roles. ...",
  "Task Family": "Frame Definition"
}
```

Training and evaluation use only `Instruction`, `Question` and `Answer`. `Task Family` is there for analysis: it lets researchers study the distribution of the supervision types or train on a selected subset. The correspondence between Task Family values and the paper's task numbers T1–T40 is in `data/README.md`.

**Load with the `datasets` library:**

```python
from datasets import load_dataset

ds = load_dataset("sag-uniroma2/FrameLLaMA-FullFN17-Instructions")
print(ds)                       # DatasetDict with 'train' and 'test' (the three test files together)
print(ds["train"][0])

# one question type of the test split
yes_no = load_dataset("sag-uniroma2/FrameLLaMA-FullFN17-Instructions",
                      data_files="test/Closed.json", split="train")

# e.g. keep only frame-relation supervision
relations = ds["train"].filter(lambda x: x["Task Family"] in {"Inherit From", "Uses", "Precedes"})
```

**No download needed.** `scripts/train.py` reads `train.json` and `scripts/evaluate_qa.py` reads the files in `test/` directly from Hugging Face (`hf://datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions/...`). The `datasets` library downloads each file once and caches it (by default in `~/.cache/huggingface/`). To keep a local copy instead, for example on a machine without internet access:

```bash
huggingface-cli download sag-uniroma2/FrameLLaMA-FullFN17-Instructions \
    --repo-type dataset --local-dir data/hf
python scripts/train.py --config configs/acl2026.yaml --train-data data/hf/train.json
```

Note the following when using these files:

- **In-domain test vs. downstream benchmarks.** The `test` split measures in-domain acquisition of FrameNet knowledge. The downstream benchmarks of the paper, NLI and SRL, are separate resources in `evaluation/splits/` (Sections 6.2–6.3).
- **Not the multilingual corpus.** [sag-uniroma2/FrameLLaMA-Multilingual-Instructions](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-Multilingual-Instructions) (English, Italian, Urdu) was built with a different generation procedure and training configuration for a separate study. Its English portion is not this corpus.
- **Supervision, not labels.** The corpus provides structured semantic supervision, not labels for any downstream benchmark.

---

## 4. Repository Structure

```text
FrameLLaMA/
├── README.md
├── LICENSE
├── requirements.txt            # inference, evaluation, dataset generation
├── requirements-train.txt      # + fine-tuning (Unsloth, TRL)
├── environment.yml             # conda environment of the evaluation runs
│
├── examples/
│   └── nli_demo.py             # one NLI example with the released model
│
├── configs/
│   ├── demo.yaml               # dataset generation on one frame (Killing)
│   └── acl2026.yaml            # all settings of the paper (generation, training, evaluation)
│
├── src/framellama/
│   ├── framenet_reader.py      # FrameNet 1.7 access and lookup tables
│   ├── templates.py            # instruction templates of the 40 task families
│   ├── negatives.py            # negative-frame and distractor filtering (SBERT similarity)
│   ├── dataset_builder.py      # generation of the 40 task families
│   ├── task_families.py        # task key -> paper task number (T1–T40)
│   ├── nli.py                  # NLI prompts, label normalisation, metrics
│   ├── srl.py, srl_prompts.py  # SRL queries, prompts, parsing, alignment
│   ├── srl_scoring_roles.py    # SRL F1: frame elements, roles, spans, partial matches
│   ├── srl_scoring_lus.py      # SRL F1: lexical units
│   ├── vllm_utils.py           # model loading (merged or LoRA) with clear errors
│   └── config.py               # config and path helpers
│
├── scripts/
│   ├── generate_dataset.py
│   ├── train.py
│   ├── merge_lora.py
│   ├── evaluate_nli.py
│   ├── evaluate_srl.py
│   └── evaluate_qa.py          # in-domain FrameNet QA on the test split
│
├── data/
│   ├── README.md               # dataset schema and provenance
│   ├── train/                  # three-record example files in the released format (offline tests)
│   ├── samples/                # a few real generated records
│   ├── manifests/acl2026.json  # facts about the released training corpus
│   ├── raw/                    # put FrameNet 1.7 here (not distributed)
│   └── generated/              # output of generate_dataset.py
│
├── evaluation/
│   ├── README.md
│   ├── prompts/                # exact NLI prompts used in the paper
│   └── splits/                 # test sets used in the paper
│       ├── nli/test_confer.json        # CONFER (2,348 problems)
│       ├── nli/test_snli.json          # SNLI diagnostic subset (9,824 pairs)
│       ├── srl/srl_gold_std.json       # SRL gold annotations (1,125 sentences)
│       └── srl/test_srl_sentences.json # SRL sentence–frame pairs (5,694)
│
├── results/acl2026/            # released predictions of the paper's runs
│
├── docs/huggingface/           # dataset card and model card (Hugging Face READMEs)
├── tools/                      # original generator, equivalence check, dataset statistics
└── tests/
```

All commands can be run from any directory: paths inside config files are resolved against the repository root.

---

## 5. Dataset Generation

The repository provides the code used to transform FrameNet into instruction-style training supervision:

```text
FrameNet 1.7
     |
     v
Frame and relation extraction                 (framenet_reader.py)
     |
     v
Task-family and template instantiation        (templates.py, dataset_builder.py)
     |
     v
Positive and negative example generation      (dataset_builder.py)
     |
     v
Negative-candidate filtering                  (negatives.py: SBERT cosine similarity < 0.50)
     |
     v
Instruction-tuning dataset                    (JSONL)
```

Each frame is generated in two passes. Sixteen task families are paraphrased: the second pass uses a second template variant with fresh distractors and both passes are kept. For the other task families only the first pass is kept.

### 5.1 Obtain FrameNet 1.7

FrameNet 1.7 can be obtained in either of two ways.

**Option A — official release.** Request the data from the [FrameNet website](https://framenet.icsi.berkeley.edu/) and unpack it so that this folder exists:

```text
data/raw/framenet_v1.7/
├── frameIndex.xml  frRelation.xml  fulltextIndex.xml  luIndex.xml  semTypes.xml
├── frame/  lu/  fulltext/
```

(The unpacked `fndata-1.7/` folder can also be placed inside `data/raw/framenet_v1.7/`.) Then pass `--framenet-dir data/raw/framenet_v1.7`.

**Option B — NLTK.** Omit `--framenet-dir` and add `--download-framenet`; the generator downloads NLTK's `framenet_v17` package, which contains the same release.

In both cases WordNet is also needed (`--download-framenet` fetches it). The generator checks the folder and reports missing files. FrameNet files are not redistributed in this repository; see the FrameNet license.

### 5.2 Generate a Small Example Dataset

```bash
python scripts/generate_dataset.py \
  --framenet-dir data/raw/framenet_v1.7 \
  --config configs/demo.yaml \
  --output data/generated/demo.jsonl
```

This runs the same generation code as the full corpus on a single frame, **Killing** (about 1,500 examples covering its frame definition, frame elements, lexical units, semantic types, annotated examples and its frame relations, e.g. *Killing* is causative of *Death*). Loading FrameNet and building the lookup tables takes a few minutes; generating the frame then takes a few seconds.

To generate other frames, list them with `--frames`:

```bash
python scripts/generate_dataset.py \
  --framenet-dir data/raw/framenet_v1.7 \
  --config configs/demo.yaml \
  --frames Commerce_sell Commerce_buy Arriving \
  --output data/generated/demo_more.jsonl
```

Add `--csv-dir DIR` to also write one CSV per frame in the original research format.

Each line of the output is one example. The first three fields have the names and meaning of the released data; the others record provenance (paper task number, frame, and the templates used):

```json
{
  "Question": "Can you describe what typically happens in the \"Killing\" frame? ...",
  "Answer": "The \"Killing\" frame represents a type of situation or event ...",
  "Task Family": "Frame Definition",
  "task_id": "T1",
  "task_key": "Frame Definition | Q1",
  "frame_name": "Killing",
  "question_id": "1",
  "question_template": "FDQ1",
  "answer_type": "Opened",
  "answer_template": "FDQ1",
  "source": "FrameNet 1.7"
}
```

A summary with the number of examples per task and a SHA-256 checksum is written to `<output>.manifest.json`. The schema is documented in `data/README.md`.

### 5.3 Generate the Full Training Dataset

```bash
python scripts/generate_dataset.py \
  --framenet-dir data/raw/framenet_v1.7 \
  --config configs/acl2026.yaml \
  --output data/generated/acl2026.jsonl
```

`configs/acl2026.yaml` sets: all 1,221 FrameNet 1.7 frames, the sentence-embedding model (`all-MiniLM-L6-v2`), the frame-level negative threshold (0.50), and the random seed. The task-level similarity thresholds (0.5 or 0.4) and the templates are part of the code.

**Relation to the released files.** To reproduce the paper, use the released dataset (Section 3.2), not a regeneration. The generator produces the questions, answers and Task Family values. The generic `Instruction` sentence of the released files was added when the per-frame output was assembled into `train.json` and the `test/` files.

**Reproducibility.** The released corpus was generated without a fixed random seed, so a new run produces a corpus with the same construction but different random choices (sampled distractors and option order). The generator code itself is unchanged from the one that produced the corpus: run on the same frames with the same seed, the reorganised code and the original research script produce byte-identical output (check it with `python tools/check_equivalence.py`). With the seed in the config, regeneration is deterministic: each frame is seeded with `<seed>:<frame name>`, so a frame produces the same examples whether it is generated alone or as part of the full run. Small differences can still come from the embedding model's floating-point results on different hardware, when a similarity lies very close to a threshold.

### 5.4 Dataset Format and Provenance

`data/README.md` documents:
- each field of the released and generated records;
- which FrameNet information each task family uses;
- how negatives are built and filtered;
- the table of the 40 task families (paper task number, Task Family value, answer type).

`data/manifests/acl2026.json` records the provenance of the released corpus. To compute the statistics of the released files (examples per split and per Task Family, checksums), run:

```bash
python tools/dataset_stats.py data/hf/train.json data/hf/test/Opened.json data/hf/test/Closed.json data/hf/test/MCQ.json
```

---

## 6. Reproducing the ACL 2026 Experiments

### 6.1 Model Adaptation

The paper uses **Meta-Llama-3.1-8B-Instruct** as the base model (Unsloth 4-bit build).

| Hyperparameter | Value |
|---|---|
| LoRA rank / alpha / dropout | 8 / 16 / 0.1 |
| LoRA target modules | q, k, v, o, gate, up, down projections |
| Batch size × gradient accumulation | 2 × 16 (effective 32) |
| Learning rate, schedule, warm-up | 5e-6, linear, 300 steps |
| Optimizer, weight decay, grad. clipping | AdamW 8-bit, 0.01, 1.0 |
| Epochs, checkpoints | 10 epochs, no validation set; checkpoint every 500 steps, final model kept |
| Max. sequence length, precision | 2048, fp16 on a 4-bit base |
| Seed | 3407 |
| Loss | on response tokens only |

All values are in the `training` section of `configs/acl2026.yaml`.

Training uses the packages from `requirements-train.txt` (installed once in Section 2.1).

```bash
python scripts/train.py --config configs/acl2026.yaml
python scripts/merge_lora.py --config configs/acl2026.yaml   # merged FP16 model for vLLM
```

`train.py` reads the training set directly from Hugging Face:

```text
hf://datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions/train.json
```

The file is downloaded on the first run and cached (`training.train_data` in the config). To train on another copy, pass `--train-data` with a local JSON/JSONL file or a URL, for example:

```bash
python scripts/train.py --config configs/acl2026.yaml \
    --train-data https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions/resolve/main/train.json
```

Only `Instruction`, `Question` and `Answer` are used. No validation set is used: training runs for the configured epochs, saves a checkpoint every 500 steps, and keeps the final model. The `test/` files are never used during training. If the run is interrupted, it resumes from the latest checkpoint. The script saves LoRA adapters, and `merge_lora.py` writes the merged FP16 model used by the evaluation scripts.

While the dataset is private, log in first with an account that has access (`huggingface-cli login`).

### 6.2 Natural Language Inference

The paper evaluates event-level inference on:

- **CONFER** (2,348 problems): `evaluation/splits/nli/test_confer.json`, and
- **an SNLI diagnostic subset** of pairs whose sentences contain FrameNet lexical units (9,824 pairs): `evaluation/splits/nli/test_snli.json`,

in zero-shot and few-shot settings (seven fixed CONFER exemplars).

```bash
python scripts/evaluate_nli.py --config configs/acl2026.yaml
```

This evaluates the base model and FrameLLaMA on both benchmarks in both settings, with the original prompts (`evaluation/prompts/`), greedy decoding, and the original label normalisation. Per-example predictions and metrics (accuracy, weighted P/R/F1, per-class scores) are written to `results/runs/nli/`. Use `--models`, `--benchmarks`, `--settings` to run a subset, `--prompt-format gpt-oss` for the gpt-oss baseline, and `--dry-run` to check data and prompts without a GPU. Details: `evaluation/README.md`.

### 6.3 Semantic Role Labeling

SRL evaluation tests sensitivity to frame and participant-role structure with three zero-shot sub-tasks over 5,694 gold annotations (1,125 sentences) in `evaluation/splits/srl/srl_gold_std.json` (`evaluation/splits/srl/test_srl_sentences.json` lists the same sentence–frame pairs):

- **roles**: given sentence, frame and lexical unit, predict the frame elements and their spans;
- **lu**: given sentence and frame, predict the lexical unit;
- **frame**: given sentence and lexical unit, predict the frame.

```bash
python scripts/evaluate_srl.py --config configs/acl2026.yaml
```

Answers are parsed as JSON and scored with the original F1 code: roles only, exact role+span, spans only, one-word overlap, partial overlap, and lexical units (exact and lemmatised). Existing files can be scored without a GPU:

```bash
python scripts/evaluate_srl.py --score-only --task roles \
    --result-file <model annotations>.json --gold-file <aligned gold>.json
```

### 6.4 In-Domain FrameNet QA

To check that the frame knowledge was acquired, the paper evaluates the base model (zero-shot) and FrameLLaMA on the `test` split of the released dataset: 247.9k held-out instances in three files.

```bash
python scripts/evaluate_qa.py --config configs/acl2026.yaml --task closed   # test/Closed.json, 81.7k yes/no
python scripts/evaluate_qa.py --config configs/acl2026.yaml --task mcq      # test/MCQ.json, 106.8k multiple choice
python scripts/evaluate_qa.py --config configs/acl2026.yaml --task opened   # test/Opened.json, 59.4k open-ended
```

The script reads the file of the requested type directly from Hugging Face (or a local copy given with `--test-file`) and prompts the models with the training (Alpaca) format and greedy decoding. It writes per-question predictions to `results/runs/qa/` and prints yes/no accuracy.

### 6.5 Reference Results

Results reported in the paper (accuracy for NLI, F1 for SRL):

| Benchmark | Setting | Base (Llama-3.1-8B-Instruct) | FrameLLaMA |
|---|---|---|---|
| CONFER | Zero-shot | 0.52 | 0.55 |
| CONFER | Few-shot | 0.39 | 0.75 |
| SNLI diagnostic subset | Zero-shot | 0.60 | 0.62 |
| SNLI diagnostic subset | Few-shot | 0.73 | 0.77 |
| SRL: lexical unit identification | Zero-shot | 0.50 | 0.62 |
| SRL: frame prediction | Zero-shot | 0.66 | 0.81 |
| SRL: frame elements, roles only | Zero-shot | 0.17 | 0.34 |
| SRL: frame elements, one-word overlap | Zero-shot | 0.13 | 0.23 |

Class-wise results and the other baselines (gpt-oss-20B, Gemma, DeepSeek-R1-Distill) are in the paper. Released prediction files are listed in `results/acl2026/README.md`.

---

## 7. Reproducibility

The repository provides:

- Source code for dataset generation, identical in behaviour to the code that produced the corpus.
- The configuration of the published corpus and experiments (`configs/acl2026.yaml`).
- The released training dataset ([Hugging Face](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions)) with its dataset card.
- The fine-tuned model checkpoint.
- Fine-tuning and merging scripts with the original settings.
- Evaluation scripts, prompts and test data.
- Prediction files of the paper's runs.

To reproduce the complete workflow:

1. Obtain FrameNet 1.7 (Section 5.1).
2. Generate a small sample dataset (Section 5.2).
3. Regenerate the full training corpus (Section 5.3), or download the released one (Section 3.2) to reproduce the paper exactly.
4. Fine-tune and merge the base model (Section 6.1).
5. Evaluate the resulting checkpoint (Sections 6.2–6.4).

Run the tests with `pytest tests`. Exact reproducibility depends on software versions, hardware and random seeds; known deviations are documented in Section 5.3 and in `evaluation/README.md`.

---

## 8. Citation

If you use FrameLLaMA, its code, or its FrameNet-derived training resources, please cite:

```bibtex
@inproceedings{rai-etal-2026-frame,
    title = "Frame-Semantic Knowledge Injection for Event-Level Inference in {LLM}s",
    author = "Rai, Shahid Iqbal  and
      Croce, Danilo  and
      Basili, Roberto",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-short.55/",
    doi = "10.18653/v1/2026.acl-short.55",
    pages = "664--678",
    ISBN = "979-8-89176-391-3"
}
```

---

## 9. License

The code in this repository is released under the [Apache License 2.0](LICENSE).

The FrameNet resource, the FrameNet-derived training dataset, the pretrained models (Llama 3.1 Community License), and the external benchmark datasets are subject to their respective licenses and terms of use. Please consult the corresponding resource providers before redistributing or modifying third-party artifacts.
