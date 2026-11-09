# Data

This folder holds the output of the dataset generator, small example files
in the released format, and a manifest of the released training corpus.

```text
data/
├── generated/               output of scripts/generate_dataset.py: one CSV per frame (not in git)
├── train/                   three-record example files in the released format (offline tests)
└── manifests/acl2026.json   facts about the released training corpus
```

## FrameNet 1.7

The generator reads FrameNet 1.7 through NLTK and downloads the packages it
needs (`framenet_v17`, `wordnet`, `omw-1.4`) the first time it runs, as the
original research script did. Behind an HTTP proxy, recent NLTK versions
refuse to download unless `NLTK_ALLOW_PROXIED_URLOPEN=1` is set.

## Record formats

### Released training data

The released files ([sag-uniroma2/FrameLLaMA-FullFN17-Instructions](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions): `train.json`,
`test/Opened.json`, `test/Closed.json`, `test/MCQ.json`) are JSON lists of records with four fields:

| Field | Type | Meaning |
|---|---|---|
| `Instruction` | str | Short generic instruction, varied across examples; carries no task information |
| `Question` | str | The FrameNet-derived question (for MCQ, the options follow the question) |
| `Answer` | str | Expected response: text, `Yes`/`No`, or the correct option |
| `Task Family` | str | Kind of FrameNet knowledge used; the prompt type of the generator (e.g. `Frame Definition`, `Frame Elements`, `Uses`) |

Training uses `Instruction`, `Question` and `Answer` (`scripts/train.py`
puts them in the `### Instruction:`, `### Input:` and `### Response:` slots
of an Alpaca-style prompt). `Task Family` is metadata for analysis; the
table below gives the paper's task numbers (T1–T40) belonging to each value.

### Generator output

`scripts/generate_dataset.py` writes one CSV file per frame (for example
`data/generated/Killing.csv`), in the format of the original research script:

| Column | Meaning |
|---|---|
| `Frame Name` | the frame |
| `Prompt Type` | the Task Family (e.g. `Frame Definition`) |
| `Question ID` | question number within the Task Family |
| `Question Sample ID` | ID of the question template used (e.g. `FDQ1`) |
| `Answer Type` | `Opened` (free text), `Closed` (Yes/No) or `MCQ` (multiple choice) |
| `Answer Sample ID` | ID of the answer template, or of the answer variant |
| `Questions` | the question |
| `Answers` | the expected response |

With `--jsonl FILE` it also writes all examples to one JSON-lines file, with
the field names of the released data plus provenance fields:

| Field | Type | Meaning |
|---|---|---|
| `Question` | str | as above |
| `Answer` | str | as above |
| `Task Family` | str | as above |
| `task_id` | str | Task number in the paper (`T1`–`T40`) |
| `task_key` | str | `"<Task Family> \| Q<question_id>"`: the generator's identifier of the task |
| `frame_name` | str | FrameNet frame the example was generated from (underscores replaced by spaces) |
| `question_id` | str | Question number within the Task Family |
| `question_template` | str | ID of the question template used (e.g. `FDQ1`, `FE5Q2`) |
| `answer_type` | str | `Opened` (free text), `Closed` (Yes/No) or `MCQ` (multiple choice) |
| `answer_template` | str | ID of the answer template, or of the answer variant |
| `source` | str | Always `FrameNet 1.7` |

The generator does not add the generic `Instruction` sentence. In the released
files it was added when the per-frame CSV files were assembled into
`train.json` and the `test/` files.

## How examples are built

For each frame, `dataset_builder.fill_template()` instantiates the templates
in `src/framellama/templates.py` with information read from FrameNet:
frame definition, frame elements (FEs) and their definitions, FE semantic
types, lexical units (LUs) and their definitions and POS, annotated example
sentences with FE spans, and frame-to-frame relations.

Lexical units are used only if WordNet lists their lemma with the same POS.

**Negatives.** Yes/No questions with answer "No", wrong MCQ options and
"which one is not…" questions use material from other frames. Two filters
keep these negatives from being accidentally correct:

1. Frame level: a frame Fn can supply negatives for a target frame Ft only if
   it is not directly related to Ft and the cosine similarity of their text
   bundles (definition, FEs, semantic types, LU definitions, relations),
   embedded with `all-MiniLM-L6-v2`, is below 0.50.
2. Candidate level: among the candidates, the first one whose similarity to
   the correct answer is below 0.5 (0.4 for some tasks) is used.

**Two passes.** Every frame is generated twice. For the 16 paraphrased task
families (`PARAPHRASED_TASKS` in `templates.py`) the second pass uses the
second template variant with new distractors and both passes are kept; for
the others only the first pass is kept.

## The 40 task families

Task numbers and descriptions follow Table 7 (Appendix E) of the paper. Each task key written by the generator was matched to a task by its answer type and question template (`src/framellama/task_families.py`). The part of the task key before `|` is the `Task Family` value.

| Task | Task key | Description |
|---|---|---|
| T1 | Frame Definition \| Q1 | Open: question about frame definition |
| T2 | Frame Definition \| Q2 | Open: detect frame from definition |
| T3 | Frame Definition \| Q3 | Yes/No: match/mismatch of frame definitions |
| T4 | Frame Elements \| Q1 | Open: FE inventory |
| T5 | Frame Elements \| Q6 | Open: definition of specific FE roles |
| T6 | Frame Elements \| Q2 | Open: detect frame from subset of FEs |
| T7 | Frame Elements \| Q5 | Yes/No: membership of FE in target frame |
| T8 | Frame Elements \| Q7 | Yes/No: validity of FE-specific definitions |
| T9 | Frame Elements \| Q3 | MCQ: contributor vs. non-contributor roles |
| T10 | Frame Elements \| Q4 | MCQ: identifying exclusive roles |
| T11 | Semantic Type \| Q1 | Open: semantic category of specific FEs |
| T12 | Semantic Type \| Q2 | Yes/No: FE-to-semantic-type mapping validity |
| T13 | Semantic Type \| Q3 | MCQ: selection of semantic type |
| T14 | Lexical Units \| Q1 | Open: LU triggers for target frame |
| T15 | Lexical Units \| Q3 | Open: functional use of LU in frame context |
| T16 | Lexical Units \| Q2 | Yes/No: LU-POS suitability for frame |
| T17 | Lexical Units \| Q4 | Yes/No: LU definition vs. frame evocation |
| T18 | Lexical Unit Polysemy \| Q1 | Open: LU polysemy to frame senses |
| T19 | Lexical Unit Polysemy \| Q2 | Yes/No: polysemous sense-to-frame linking |
| T20 | Lexical Units \| Q5 | MCQ: discriminating between frame senses |
| T21 | Frame Elements Examples \| Q1 | Open: example sentences for specific FEs |
| T22 | Frame Elements Examples \| Q2 | Yes/No: FE role binding in annotated examples |
| T23 | Inherit From \| Q1 | Open: inheritance, parent frame |
| T24 | Inherit From \| Q2 | Yes/No: inheritance link validity |
| T25 | Inherit from \| Q3 | MCQ: inheritance, child from parent (lower-case "from" as in the code) |
| T26 | Perspective On \| Q1 | Open: alternative viewpoint frame |
| T27 | Perspective On \| Q2 | Yes/No: viewpoint link verification |
| T28 | Perspective On \| Q3 | MCQ: discriminating viewpoints |
| T29 | Uses \| Q1 | Open: structural foundation frame |
| T30 | Uses \| Q2 | Yes/No: reused concept verification |
| T31 | Uses \| Q3 | MCQ: base frame discrimination |
| T32 | SubFrame of \| Q1 | Open: component event identification |
| T33 | SubFrame of \| Q2 | Yes/No: part-of relation validity |
| T34 | Precedes \| Q1 | Open: temporal sequence ordering |
| T35 | Precedes \| Q2 | Yes/No: sequential link verification |
| T36 | Is Causative of \| Q1 | Open: causative, resulting state |
| T37 | Is Causative of \| Q2 | Yes/No: causative link validity |
| T38 | Is Inchoative of \| Q1 | Open: inchoative, resulting state |
| T39 | Is Inchoative of \| Q2 | Yes/No: inchoative outcome verification |
| T40 | Is Causative of \| Q3 | MCQ: discriminating outcomes (see note below) |

The paper lists T36–T37 as Causative and T38–T40 as Inchoative. In the code, the MCQ task of this group (T40) is generated from the *Is Causative of* relation, so the Causative group has three tasks (T36, T37, T40) and the Inchoative group two (T38, T39).

## Released training corpus

**https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions**

| Split | File | Use in the paper |
|---|---|---|
| train | `train.json` | more than 491K instances: LoRA fine-tuning |
| test | `test/Opened.json` | 59.4k open-ended: in-domain FrameNet QA evaluation |
| | `test/Closed.json` | 81.7k yes/no |
| | `test/MCQ.json` | 106.8k multiple choice |

These are the exact files of the published training run. The scripts read
them directly from Hugging Face (`training.train_data` and
`evaluation.qa.test_file` in `configs/acl2026.yaml`); nothing needs to be
downloaded by hand. For a local copy:

```bash
huggingface-cli download sag-uniroma2/FrameLLaMA-FullFN17-Instructions \
    --repo-type dataset --local-dir data/hf
```

Statistics of the files (examples per split and per Task Family, SHA-256):

```bash
python tools/dataset_stats.py data/hf/train.json data/hf/test/*.json
```

`data/manifests/acl2026.json` records the provenance of the corpus.

The `test` split measures in-domain acquisition of FrameNet knowledge
(`scripts/evaluate_qa.py`); the downstream NLI and SRL benchmarks are
separate resources in `evaluation/splits/`. Each test file holds one
question type. The
corpus is also distinct from the English portion of
[FrameLLaMA-Multilingual-Instructions](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-Multilingual-Instructions),
which was built with a different procedure for a separate study.

The released corpus was generated without a fixed seed, so regenerating it
gives a corpus built in the same way but with different random choices; it
will not match the released files line by line. Use the released files to
reproduce the paper.

## A small sample

To get a few records per task family of one frame in the JSON-lines format:

```bash
python scripts/generate_dataset.py --frames Killing --jsonl data/generated/killing.jsonl
python tools/make_sample.py data/generated/killing.jsonl data/generated/sample.jsonl --per-task 2
```
