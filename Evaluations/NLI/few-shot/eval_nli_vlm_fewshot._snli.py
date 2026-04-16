import os
import csv
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# ================================
# CONFIG
# ================================
#MODEL_PATH = "merged_finetuned_llama3.1_8b_fp16_vllm_full"
#MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_PATH = "google/gemma-3-4b-it"
OUTPUT_CSV = "Evaluation/Pred/NLI/baseline_nli_few_1.csv"
TEST_FILE = "Evaluation/NLI/snli_matched_FrameNet.json"

os.makedirs("Evaluation/Pred/NLI", exist_ok=True)

LABELS = {"E", "C", "N"}

def normalize_label(x):
    x = x.lower().strip()
    x = x.replace(".", "").replace("!", "").replace("?", "").replace("</s>", "")
    for label in LABELS:
        if label in x:
            return label
    return "neutral"  # fallback

def write_to_csv(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Premise", "Hypothesis", "Predictions", "References"])
        writer.writerows(rows)


def compute_similarity_score_sbert(references, predictions):
    model = SentenceTransformer("all-MiniLM-L12-v2")
    emb_ref = model.encode(references, convert_to_tensor=True)
    emb_pred = model.encode(predictions, convert_to_tensor=True)
    sims = [1 - cosine(r.cpu().numpy(), p.cpu().numpy())
            for r, p in zip(emb_ref, emb_pred)]
    return sum(sims) / len(sims)


def generate_predictions(llm, dataset):
    prompts = []
    gold = []
    csv_rows = []
    predictions = []
    references = []

    for sample in dataset:
        premise = sample["premise"].strip()
        hypothesis = sample["hypothesis"].strip()
        prompt = f"""
        # Task: Judge if the hypothesis necessarily follows from the premise.
        # Consider the truth value of the premise. If the premise is true, does it necessarily mean that the hypothesis must also be true?

        # Output:
        E — choose this option if the hypothesis is necessarily true in every case.
        C — choose this option if the hypothesis is always untrue and cannot be correct.
        N — choose this option if the hypothesis can be true in some situations and false in others.

        ### Expected Outputs:

        Premise: If Emily passes her driving test, she'll never read a book by Dan Brown again.  
        Hypothesis: Emily has read a book by Dan Brown before.  
        Output: E

        Premise: If Margaret's flight arrives on time, her friend will be waiting for her at the airport.  
        Hypothesis: Margaret has a friend.  
        Output: E

        Premise: Sam believes that if the boss rewarded a good behavior by Nava, they'll never reward a good behavior by Nava again.  
        Hypothesis: The boss rewarded a good behavior by Nava.  
        Output: N

        Premise: If Alexander has a bright roommate, he'll rest with his bright roommate.  
        Hypothesis: Alexander has a bright roommate.  
        Output: N

        Premise: It's not the case that if Curtis reports on this issue, he'll never report on an issue again.  
        Hypothesis: Curtis has reported on an issue before.  
        Output: N

        Premise: Sam believes that if Alice ever drank a cocktail, she didn't drink martini again.  
        Hypothesis: Alice has drunk a martini.  
        Output: N

        Premise: If Meredith has video games, she will play her immersive video game.  
        Hypothesis: Meredith has an immersive video game.  
        Output: N

        # Do not output anything other than letters N, C, or E.

        Premise: {premise}
        Hypothesis: {hypothesis}
        # Output:"""
        prompts.append(prompt)
        gold.append(sample["gold_label"])

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=1,
        #top_k=10,
        stop=["</s>"]
    )

    outputs = llm.generate(prompts, sampling_params)

    for i, out in enumerate(outputs):
        pred = out.outputs[0].text.strip()
        ref = gold[i].strip()

        predictions.append(pred)
        references.append(ref)

        csv_rows.append([
            dataset[i]["premise"],
            dataset[i]["hypothesis"],
            pred,
            ref
        ])

    write_to_csv(csv_rows)
    return predictions, references


# ================================
# MAIN LOGIC (NEEDED FOR VLLM)
# ================================
if __name__ == "__main__":

    print("🔵 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("🚀 Loading model with vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        #dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=8192,
    )

    print("📘 Loading SNLI dataset...")
    dataset = load_dataset("json", data_files=TEST_FILE)["train"]
    print("Dataset size:", len(dataset))

    print("⚡ Running inference...")
    predictions, references = generate_predictions(llm, dataset)

    print("\n=== SAMPLE OUTPUTS ===")
    for i in range(10):
        print("REF:", references[i])
        print("PRED:", predictions[i])
        print("-" * 40)

    # Compute metrics
    print("\n📊 Evaluation Results:")
    print("Accuracy :", accuracy_score(references, predictions))
    print("F1 Score :", f1_score(references, predictions, average="weighted"))
    print("Recall   :", recall_score(references, predictions, average="weighted"))
    print("Precision:", precision_score(references, predictions, average="weighted"))

    # Additional accuracy using simple equality
    simple_acc = np.mean(np.array(predictions) == np.array(references))
    print("Accuracy (simple np.mean)  :", simple_acc)

    print("\n🔢 Reference label distribution:")
    print(pd.Series(references).value_counts().to_string())

    print("\n🔢 Prediction label distribution:")
    print(pd.Series(predictions).value_counts().to_string())

    # Optional SBERT similarity
    sim_score = compute_similarity_score_sbert(references, predictions)
    print(f"\n🔵 SBERT Similarity Mean: {sim_score:.4f}")
