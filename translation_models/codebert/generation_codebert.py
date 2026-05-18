import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ==========================
# SETTINGS
# ==========================
MODEL_NAME = "microsoft/codebert-base"
DATA_JSONL = "/workspace/combined.jsonl"
OUTPUT_DIR = "/workspace/codebert_java_to_csharp"

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
# PROMPT FORMAT
# ==========================
def build_example(ex):

    return {
        "input_text": f"translate Javacode  to equivalent  C# code : {ex['source_code']}",
        "target_text": ex["target_code"]
    }

train_ds = train_ds.map(build_example, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(build_example, remove_columns=eval_ds.column_names)

# ==========================
# TOKENIZER
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================
# TOKENIZATION
# ==========================
def preprocess_function(examples):

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        examples["target_text"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    model_inputs["labels"] = labels

    return model_inputs


train_tok = train_ds.map(preprocess_function, batched=True)
eval_tok = eval_ds.map(preprocess_function, batched=True)

# ==========================
# MODEL
# ==========================
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    MODEL_NAME,
    MODEL_NAME
)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id

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
    learning_rate=3e-5,
    num_train_epochs=5,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    predict_with_generate=True,
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

print("Starting CodeBERT Fine-Tuning...")
trainer.train()

# ==========================
# SAVE MODEL
# ==========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete.")
print("Model saved to:", OUTPUT_DIR)
