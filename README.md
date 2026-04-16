# 🧠 Frame Semantic Knowledge Injection for Event-Level Inference in LLMs

This repository contains the official implementation and resources for
our ACL 2026 paper:\
**"Frame Semantic Knowledge Injection for Event-Level Inferences in
LLMs."**

------------------------------------------------------------------------

## 📌 Overview

Large Language Models (LLMs) are powerful but often struggle with
**event-level reasoning**, especially when it depends on **semantic
roles, event structures, and inter-event relations**.

This work injects **FrameNet-based semantic knowledge** into LLMs using
**LoRA fine-tuning**, improving reasoning across:

-   Natural Language Inference (NLI)\
-   Semantic Role Labeling (SRL)\
-   Question Answering (QA)

------------------------------------------------------------------------

## 📂 Repository Structure

    ├── Datasets/
    │   ├── train/
    │   ├── test/
    │   │   ├── NLI/
    │   │   ├── SRL/
    │   │   └── QA/
    │
    ├── Evaluation/
    │   ├── NLI
    │   ├── SRL
    │   ├── QA
    │
    │
    ├── Results/
    │   ├── NLI/
    │   ├── SRL/
    │   └── QA/
    │
    └── README.md

------------------------------------------------------------------------

## 🚀 Getting Started

### Setup

    git clone https://github.com/shahidrai/frame-aware-nli.git
    cd frame-aware-nli
    pip install -r requirements.txt

------------------------------------------------------------------------

## 📊 Datasets

-   **Training**: 491k+ QA pairs from FrameNet\
-   **NLI**: CONFER + SNLI subset\
-   **SRL**: OpenSesame\
-   **QA**: FrameNet-based

------------------------------------------------------------------------

## 🧪 Evaluation

### NLI

    python Evaluation/nli_inference.py --model_path <model> --dataset_path Datasets/test/NLI/

### SRL

    python Evaluation/srl_inference.py --model_path <model> --dataset_path Datasets/test/SRL/

### QA

    python Evaluation/qa_inference.py --model_path <model> --dataset_path Datasets/test/QA/

------------------------------------------------------------------------

## 📈 Results

-   Improved **NLI accuracy**, especially entailment & contradiction\
-   Strong gains in **SRL F1 scores**\
-   Better **event-level reasoning and role understanding**

------------------------------------------------------------------------

## 🎯 Contributions

-   FrameNet-based supervision (1200+ frames)\
-   491k QA dataset\
-   LoRA-based efficient fine-tuning\
-   Transfer to NLI and SRL


------------------------------------------------------------------------

## 📜 Citation

    @inproceedings{acl2026,
      title={Frame Semantic Knowledge Injection for Event-Level Inferences in LLMs},
      author={Iqbal, Shahid and Croce, Danilo and Basili, Roberto},
      booktitle={ACL 2026},
      year={2026}
    }


------------------------------------------------------------------------

## ⭐ Star this repo if you find it useful!

