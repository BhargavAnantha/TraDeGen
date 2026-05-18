#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

# ==================== Settings ====================
MODEL_NAME = "codellama/CodeLlama-7b-hf"
DATA_JSONL = "/workspace/combined.jsonl"
OUTPUT_DIR = "/workspace/codellama_java_to_csharp"

# ==================== Load Dataset ====================
dataset = load_dataset("json", data_files=DATA_JSONL, split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# ==================== Load Tokenizer ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # Llama needs a pad token defined

# ==================== Prompt Formatting ====================
def format_instruction(ex):
    # Causal models need the prompt and answer in one string
    full_prompt = (
        f"### Instruction: Translate the following Java code to C#.\n"
        f"### Java:\n{ex['source_code']}\n"
        f"### C#:\n{ex['target_code']}{tokenizer.eos_token}"
    )
    return {"text": full_prompt}

train_ds = dataset["train"].map(format_instruction)
eval_ds = dataset["test"].map(format_instruction)

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=512)

train_tok = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
eval_tok = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names)

# ==================== Load Model (4-bit) ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], # Standard for Llama
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ==================== Training Args ====================
train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4, # Higher LR is common for LoRA
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    report_to="none"
)

# ==================== Trainer ====================
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting CodeLlama Finetuning...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
