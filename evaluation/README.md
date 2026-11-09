# Evaluation data and protocol

```text
evaluation/
├── prompts/                         exact prompts of the paper's NLI runs
│   ├── nli_zero_shot_confer.txt
│   ├── nli_zero_shot_snli.txt
│   ├── nli_few_shot_confer.txt      7 fixed CONFER exemplars
│   ├── nli_few_shot_snli.txt        same exemplars, SNLI wording
│   ├── nli_gptoss_system.txt        gpt-oss baseline (harmony format)
│   ├── nli_zero_shot_gptoss_developer.txt
│   └── nli_few_shot_gptoss_developer.txt
└── splits/
    ├── nli/test_confer.json         CONFER, 2,348 problems (premise, hypothesis, gold_label E/C/N)
    ├── nli/test_snli.json           SNLI diagnostic subset, 9,824 pairs (sentence1, sentence2, gold_label)
    ├── srl/srl_gold_std.json        1,125 sentences, 5,694 frame annotations with FE spans
    └── srl/test_srl_sentences.json  the same 5,694 (sentence, frame) pairs, detokenised; not used by the scripts
```

The prompt files were extracted programmatically from the original
evaluation scripts, so their text (including whitespace) is exactly what was
sent to the models. SRL prompts are in `src/framellama/srl_prompts.py`,
copied from the original SRL scripts.

## Data sources

* **CONFER** (`nli/test_confer.json`): the 2,348 inference problems of the CONFER benchmark.
* **SNLI diagnostic subset** (`nli/test_snli.json`): 9,824 premise–hypothesis pairs from SNLI used in the paper's diagnostic evaluation. SNLI is distributed under CC BY-SA 4.0.
* **SRL gold standard** (`srl/srl_gold_std.json`): 1,125 sentences with 5,694 gold annotations (frame, lexical unit, frame-element spans); `srl/test_srl_sentences.json` lists the same sentence–frame pairs.

These resources remain subject to the terms of their original distributions.

## NLI protocol (`scripts/evaluate_nli.py`)

| Setting | Prompt | Decoding | Label mapping |
|---|---|---|---|
| zero-shot | `nli_zero_shot_<benchmark>.txt` | greedy, top_p 0.9, max 2 tokens, stop at newline | output and gold mapped to entailment / contradiction / neutral with `normalize_model_output` / `normalize_reference_label` |
| few-shot | `nli_few_shot_<benchmark>.txt` | greedy, top_p 0.9, max 1 token | generated letter compared with the gold letter E/C/N |
| gpt-oss | system + developer message (harmony) | greedy, top_p 0.9, max 2000 tokens | last `assistantfinal{E,C,N}` in the output |

Prompts are given as plain text (no chat template), as in the original runs.
Metrics: accuracy, weighted precision/recall/F1 and per-class scores
(per-class recall is the class-wise accuracy reported in the paper).

## SRL protocol (`scripts/evaluate_srl.py`)

One query per gold annotation (5,694 per task), greedy decoding, up to 512
tokens, stop at ```` ``` ````. The answer is parsed as JSON with the original
parsers. For each sentence, the predicted annotations are written to
`<run>_result.json` and the gold annotations of the same queries to
`<run>_gold-std.json`; queries whose answer could not be parsed are left out
of both files. The two files are then scored with the original F1 code:

| Metric | Definition |
|---|---|
| Lexical units | F1 of (frame, LU) pairs; also after lemmatisation |
| Frame elements | F1 of exact (role, span) pairs, averaged per frame then per sentence |
| Roles only | F1 of role labels |
| Spans | F1 of exact span strings |
| One-word overlap | a (role, span) pair matches if the roles are equal and the spans share a word |
| 25% matched | roles equal and at least 25% of the gold span's words in the predicted span |

## Differences from the original scripts

The original scripts had hard-coded paths to files that are not in the
repository and some errors that stopped them from running. The following
was changed; the prompts, decoding settings, parsers and metric code were not.

1. **Paths and models** come from `configs/acl2026.yaml` or the command line.
2. **SNLI fields.** `test_snli.json` uses `sentence1`/`sentence2`; the original
   SNLI scripts read `premise`/`hypothesis` and would have sent empty
   sentences. Both field names are now accepted.
3. **Few-shot gold labels.** The few-shot scripts compared the generated
   letter with the raw gold label. This works for CONFER (E/C/N) but gives 0
   on SNLI (`entailment`, …); the gold label is now converted to its letter.
4. **gpt-oss scripts** referred to an undefined `DEVELOPER_MESSAGE_1`; the
   `DEVELOPER_MESSAGE` defined in each script is used. They also wrote the
   last premise into every CSV row; each row now has its own premise.
5. **SRL inputs and alignment** (`input_sentences_combined.json` and the
   grouping into `*_result_*` / `*_gold-std_*` files) were not in the
   repository and were rebuilt from `srl_gold_std.json`. The rebuilt gold
   files have the same structure as the released `*_gold-std_*.json` files
   (order-preserving subsets of the gold annotations). Spans returned as lists
   are joined into one string so that the scorer does not fail.
6. **Frame prediction** has no separate scorer in the original code; it is
   scored with the lexical-unit scorer, which compares (frame, LU) pairs.
7. **QA** prompts are sent in one batch instead of one at a time.
8. The SBERT "similarity" between label strings printed by the NLI scripts
   was dropped; it was not a reported metric.

With the baseline predictions, the rebuilt alignment gives a roles-only F1 of 0.171, matching the 0.17 reported in the paper.
