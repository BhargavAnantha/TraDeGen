#!/usr/bin/env python3

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ==========================
# SETTINGS
# ==========================
MODEL_NAME = "uclanlp/plbart-base"
DATA_JSONL = "/workspace/Hallucination_aware_dataset.jsonl"   # ✅ NEW DATASET
OUTPUT_DIR = "/workspace/plbart_hallucination_corrector"  # ✅ NEW PATH

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512

# ==========================
# LOAD DATASET
# ==========================
dataset = load_dataset("json", data_files=DATA_JSONL, split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)

train_ds = dataset["train"]
eval_ds = dataset["test"]

# ==========================
# LOAD TOKENIZER
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================
# PREPROCESS FUNCTION (UPDATED)
# ==========================
def preprocess_function(examples):

    # ✅ NEW PROMPT (hallucination-aware)
    inputs = [
        f"Language: {lang}\n"
        f"Fix the hallucinated code based on the given hallucination  type.\n"
        f"Type: {label}\n"
        f"Code:\n{code}"
        for lang, label, code in zip(
            examples["language"],
            examples["label"],
            examples["hallucinated_code"]
        )
    ]

    targets = examples["original_code"]

    # Tokenize input
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length"
    )

    # Tokenize target
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    # Replace padding tokens with -100
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    model_inputs["labels"] = labels

    return model_inputs


train_tok = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
eval_tok = eval_ds.map(preprocess_function, batched=True, remove_columns=eval_ds.column_names)

# ==========================
# LOAD MODEL
# ==========================
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# ==========================
# DATA COLLATOR
# ==========================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# ==========================
# TRAINING ARGUMENTS
# ==========================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=5,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    predict_with_generate=True,
    max_grad_norm=1.0,
    warmup_steps=500,
    fp16=False,
    report_to="none"
)

# ==========================
# TRAINER
# ==========================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Starting PLBART Hallucination Correction Training...")
trainer.train()

# ==========================
# SAVE MODEL
# ==========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete.")
print("📦 Model saved at:", OUTPUT_DIR)
