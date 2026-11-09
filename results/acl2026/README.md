# Released predictions of the ACL 2026 runs

| File | Content |
|---|---|
| `nli/baseline_nli_0shot_confer.csv` | Llama-3.1-8B-Instruct, CONFER, zero-shot (Premise, Hypothesis, Predictions, References) |
| `nli/adapted_nli_0shot_confer.csv` | FrameLLaMA, CONFER, zero-shot |
| `srl/baseline_srl_result_roles.json` | Llama-3.1-8B-Instruct, frame-element predictions (979 sentences) |
| `srl/adapted_srl_gold-std_roles.json` | gold annotations aligned with FrameLLaMA's frame-element predictions (1,108 sentences) |
| `srl/adapted_srl_gold-std_LUs.json` | gold annotations aligned with FrameLLaMA's lexical-unit predictions (1,113 sentences) |
| `qa/baseline_closed.csv` | Llama-3.1-8B-Instruct, held-out closed (Yes/No) FrameNet questions (15,000) |

Scoring commands for these files (no GPU needed) are in README section 6.6:

```bash
python scripts/evaluate_nli.py --score-file results/acl2026/nli/baseline_nli_0shot_confer.csv results/acl2026/nli/adapted_nli_0shot_confer.csv
python scripts/evaluate_srl.py --score-only --task roles --result-file results/acl2026/srl/baseline_srl_result_roles.json
python scripts/evaluate_qa.py --score-file results/acl2026/qa/baseline_closed.csv
```

New runs of `scripts/evaluate_*.py` are written to `results/runs/`.
