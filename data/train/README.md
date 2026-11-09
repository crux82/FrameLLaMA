# Example training files

`train.json` and `test/Opened.json` here contain **three real records** in the
released format. They are used by the tests and show the format offline;
they are not the training data.

The full files are on Hugging Face and the scripts read them from there
directly (no download needed):

**https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions**

| File | Read by | Content |
|---|---|---|
| `train.json` | `scripts/train.py` | more than 491K training instances |
| `test/Opened.json`, `test/Closed.json`, `test/MCQ.json` | `scripts/evaluate_qa.py` | 247.9k held-out instances (59.4k open-ended, 81.7k yes/no, 106.8k multiple choice) for the in-domain FrameNet QA evaluation; not used in training |

Format: a JSON list of objects with the fields `Instruction`, `Question`,
`Answer`, `Task Family` (documented in `data/README.md`). Training uses only
the first three; `Task Family` is metadata for analysis.
