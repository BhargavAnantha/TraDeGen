#!/usr/bin/env python3
import os

# ==================== Environment ====================
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

# ==================== Paths ====================
DATA_JSONL = "/workspace/java_c#_parallel_codes_Translation"
MODEL_NAME = "Salesforce/codet5p-770m"
CACHE_DIR = "/workspace/.cache/huggingface"
OUTPUT_DIR = "/workspace/codet5_java_to_csharp"

# ==================== Load dataset ====================
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_JSONL, split="train")

dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# ==================== Load tokenizer ====================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# ==================== Prompt Format ====================
def build_example(ex):
    """
    Encoder input: Java
    Decoder target: C#
    """
    return {
        "input_text": f"translate Java to C#: {ex['source_code']}",
        "target_text": ex["target_code"]
    }

train_ds = train_ds.map(build_example, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(build_example, remove_columns=eval_ds.column_names)

# ==================== Tokenization ====================
MAX_SRC_LEN = 512
MAX_TGT_LEN = 512

def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SRC_LEN
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_TGT_LEN
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(tokenize_fn, batched=True)
eval_tok  = eval_ds.map(tokenize_fn, batched=True)

# ==================== Load Model ====================
print("Loading CodeT5 model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)
model.gradient_checkpointing_enable()
model.config.use_cache=False
model.to("cuda")

# ==================== Training Args ====================
train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
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

# ==================== Train ====================
print("Training CodeT5...")
trainer.train()

# ==================== Save ====================
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved to:", OUTPUT_DIR)
