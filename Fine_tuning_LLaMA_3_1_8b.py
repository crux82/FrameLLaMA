import os
import torch
import random
import numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ======================================================
# ✅ Environment Configuration
# ======================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Fix random seeds for reproducibility
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================
# ✅ Configuration
# ======================================================
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
output_dir = "FrameNet_15"
max_seq_length = 2048
load_in_4bit = True

# ======================================================
# ✅ Load Model + Tokenizer
# ======================================================
print("\n🚀 Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
EOS_TOKEN = tokenizer.eos_token

# ======================================================
# ✅ Add LoRA Adapters
# ======================================================
print("⚙️ Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ======================================================
# ✅ Alpaca-style Instruction Template
# ======================================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    """
    Converts dataset rows into Alpaca-style prompts.
    Compatible with DataCollatorForCompletionOnlyLM.
    """
    questions = examples.get("Question", None)
    answers = examples.get("Answer", None)
    inputs = examples.get("Instruction", [""] * len(questions))  # optional field
    texts = []

    for inp, q, a in zip(inputs, questions, answers):
        prompt = alpaca_prompt.format(inp, q, a) + EOS_TOKEN
        texts.append(prompt)

    return texts  # ✅ Return list directly (not a dict)


def formatting_answers_only(dataset):
    """Used for evaluation dataset (answers only)."""
    return [f"{a} {EOS_TOKEN}" for a in dataset["Answer"]]

# ======================================================
# ✅ Load Dataset
# ======================================================
print("\n📂 Loading datasets...")
dataset_train = load_dataset("json", data_files={"train": "Dataset_1/JSON/train_15.json"})["train"]
dataset_dev = load_dataset("json", data_files={"dev": "Dataset_1/JSON/val_15.json"})["dev"]

# Optional evaluation preparation
eval_texts = formatting_answers_only(dataset_dev)
eval_dataset = tokenizer(eval_texts, truncation=True, padding="max_length",
                         max_length=max_seq_length, return_tensors="pt")

# ======================================================
# ✅ Data Collator
# ======================================================
response_template="### Response:\n",
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)




# Build one formatted example (first row)
sample = formatting_prompts_func(dataset_train[:1])[0]
print("----- SAMPLE PROMPT -----")
print(sample[:800])
print("Marker present? ->", "### Response:\n" in sample)
print("-------------------------")

marker = "### Response:\n"
enc_full   = tokenizer(sample, add_special_tokens=False).input_ids
enc_marker = tokenizer(marker, add_special_tokens=False).input_ids

def contains(sub, sup):
    # naive subsequence check
    L = len(sub)
    return any(sup[i:i+L] == sub for i in range(len(sup)-L+1))

print("Marker tokens found in prompt tokens? ->", contains(enc_marker, enc_full))

# ======================================================
# ✅ Training Arguments
# ======================================================
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=300,
    num_train_epochs=10,
    learning_rate=5e-6,
    fp16=True,
    bf16=False,
    logging_steps=500,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    output_dir=output_dir,
    report_to="none",
    seed=3407,
)

# ======================================================
# ✅ Trainer Setup
# ======================================================
print("\n🏋️ Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_dev,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# ======================================================
# ✅ Resume from Latest Checkpoint
# ======================================================
checkpoint_dir = None
if os.path.isdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        checkpoint_dir = checkpoints[-1]
        print(f"🔁 Resuming from checkpoint: {checkpoint_dir}")

# ======================================================
# ✅ Train Model
# ======================================================
print("\n🚀 Starting training...")
trainer.train(resume_from_checkpoint=checkpoint_dir)

# ======================================================
# ✅ Save Fine-Tuned Model
# ======================================================
print("\n💾 Saving fine-tuned model...")
model.save_pretrained("All_FrameNet_LLaMA_Alpaca")
tokenizer.save_pretrained("All_FrameNet_LLaMA_Alpaca")

print("\n✅ Training complete! Model saved successfully.")
