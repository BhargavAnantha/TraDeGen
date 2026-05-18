#!/usr/bin/env python3
import os

# ==================== Environment ====================
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable Apex
os.environ["USE_APEX"] = "0"
os.environ["USE_TORCH"] = "1"

# ==================== Imports ====================
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# ==================== Paths ====================
DATA_JSONL = "/workspace/Hallucination_aware_dataset.jsonl"
MODEL_NAME = "Salesforce/codet5p-770m"
CACHE_DIR = "/workspace/.cache/huggingface"
OUTPUT_DIR = "/workspace/codet5_hallucination_corrector"

# ==================== Load dataset ====================
print("📂 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_JSONL, split="train")

dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# ==================== Load tokenizer ====================
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# ==================== Prompt Format ====================
def build_example(ex):
    return {
        "input_text": (
            f"Language: {ex['language']}\n"
            "Correct the following code based on the hallucination type.\n"
            f"Type: {ex['label']}\n"
            f"Code:\n{ex['hallucinated_code']}"
        ),
        "target_text": ex["original_code"]
    }

train_ds = train_ds.map(build_example, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(build_example, remove_columns=eval_ds.column_names)

# ==================== Tokenization ====================
MAX_LEN = 512

def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    # Mask padding tokens
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels_ids
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs

print("🔧 Tokenizing dataset...")
train_tok = train_ds.map(tokenize_fn, batched=True)
eval_tok  = eval_ds.map(tokenize_fn, batched=True)

# ==================== Load Model ====================
print("🤖 Loading CodeT5 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Training Arguments ====================
train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=False,
    max_grad_norm=1.0,
    warmup_steps=500,
    logging_steps=50,

    # Checkpointing (VERY IMPORTANT)
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    load_best_model_at_end=True,
    report_to="none",
)

# ==================== Trainer ====================
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer
)

# ==================== Resume Logic ====================
print("🚀 Starting training...")

checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoint = get_last_checkpoint(OUTPUT_DIR)

if checkpoint:
    print(f"🔁 Resuming from checkpoint: {checkpoint}")
else:
    print("🆕 Starting fresh training")

trainer.train(resume_from_checkpoint=checkpoint)

# ==================== Save ====================
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model saved to:", OUTPUT_DIR)
