---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
datasets:
- sag-uniroma2/FrameLLaMA-FullFN17-Instructions
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- lora
- framenet
- frame-semantics
- natural-language-inference
- semantic-role-labeling
---

# FrameLLaMA-3.1-8B-Instruct-FullFN17

Llama-3.1-8B-Instruct adapted with frame-semantic knowledge from FrameNet 1.7, from the ACL 2026 paper
**[Frame-Semantic Knowledge Injection for Event-Level Inference in LLMs](https://aclanthology.org/2026.acl-short.55/)**
(Shahid Iqbal Rai, Danilo Croce, Roberto Basili).

- **Code** (data generation, training, evaluation): https://github.com/crux82/FrameLLaMA
- **Training data:** https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions

## Model details

| | |
|---|---|
| Base model | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), trained on its 4-bit Unsloth build |
| Checkpoint | LoRA adapters merged into the base model (16-bit weights) |
| Training data | more than 491,000 instruction–response pairs generated from FrameNet 1.7, 40 task families ([sag-uniroma2/FrameLLaMA-FullFN17-Instructions](https://huggingface.co/datasets/sag-uniroma2/FrameLLaMA-FullFN17-Instructions)) |
| Adaptation | LoRA, r = 8, α = 16, dropout 0.1, on all attention and MLP projections (Unsloth) |
| Training | 10 epochs, learning rate 5e-6, effective batch size 32, AdamW 8-bit, no validation set |
| Training prompt | Alpaca format (`### Instruction:` / `### Input:` / `### Response:`) |

The training data cover frame definitions, frame elements and their semantic types, lexical units and polysemy, annotated examples, and frame-to-frame relations (inheritance, perspective, uses, subframe, precedes, causative, inchoative), with filtered negative examples.

## Intended use

Research on event-level semantics: natural language inference involving event structure and participant roles, frame identification, and semantic role labeling. It is not intended as a general-purpose assistant; the adaptation targets frame-semantic competence, not overall performance.

## Usage

The model is used with plain-text prompts (no chat template), as in the paper. Zero-shot NLI with the prompt used in the evaluation:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sag-uniroma2/FrameLLaMA-3.1-8B-Instruct-FullFN17"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="auto")
model.eval()

PROMPT = """Judge if the hypothesis necessarily follows from the premise.
Consider the truth value of the premise. If the premise is true, does it necessarily mean that the hypothesis must also be true?
Output E if the hypothesis must always be true.
Output C if the hypothesis must always be false.
Output N if the hypothesis may be either true or false.
Do not output anything other than letters E, C, or N.

Premise: {premise}
Hypothesis: {hypothesis}
# Output:"""

premise = "John drowned Martha."
hypothesis = "Martha died."

inputs = tokenizer(PROMPT.format(premise=premise, hypothesis=hypothesis), return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)

answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
label = {"E": "entailment", "C": "contradiction", "N": "neutral"}.get(answer[:1].upper(), "unknown")
print(f"Raw output: {answer!r} -> {label}")
```

For fast batch inference with vLLM, the full evaluation scripts, and the few-shot and SRL prompts, see the [GitHub repository](https://github.com/crux82/FrameLLaMA).

## Results

From the paper (accuracy for NLI, F1 for SRL; base = Llama-3.1-8B-Instruct, same prompts):

| Benchmark | Setting | Base | FrameLLaMA |
|---|---|---|---|
| CONFER | zero-shot | 0.52 | 0.55 |
| CONFER | few-shot | 0.39 | 0.75 |
| SNLI diagnostic subset | zero-shot | 0.60 | 0.62 |
| SNLI diagnostic subset | few-shot | 0.73 | 0.77 |
| SRL: lexical unit identification | zero-shot | 0.50 | 0.62 |
| SRL: frame prediction | zero-shot | 0.66 | 0.81 |
| SRL: frame elements, roles only | zero-shot | 0.17 | 0.34 |

Gains are largest for entailment on CONFER and for frame and role prediction in SRL; neutral accuracy on CONFER zero-shot decreases.

## Limitations

- Trained only on English FrameNet 1.7; coverage is limited to FrameNet frames and lexical units.
- The training data are generated from templates, so the model is most reliable with prompts close to those formats.
- It inherits the limitations and biases of Llama-3.1-8B-Instruct and is not intended for production or safety-critical use.

## License

Use is subject to the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE). The training data are derived from FrameNet 1.7; see the FrameNet license terms.

## Citation

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
