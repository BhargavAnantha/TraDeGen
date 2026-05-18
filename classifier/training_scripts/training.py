import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# SETTINGS
# =========================
MODEL_NAME = "microsoft/graphcodebert-base"
DATA_FILE = "/content/sample_data/converted_dataset.jsonl"
OUTPUT_DIR = "/content/graphcodebert_hallucination_classifier"

MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 5

# =========================
# LOAD & PREPARE DATASET
# =========================
dataset = load_dataset("json", data_files=DATA_FILE)["train"]
dataset = dataset.train_test_split(test_size=0.1)

# Get labels and create mappings
unique_labels = sorted(list(set(dataset["train"]["label"])))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Combine original and hallucinated code into one string for the model to compare
    texts = [
        f"Original Code:\n{orig}\n\nHallucinated Code:\n{hallu}"
        for orig, hallu in zip(examples["original_code"], examples["hallucinated_code"])
    ]

    # Tokenize the text
    result = tokenizer(texts, truncation=True, max_length=MAX_LEN)

    # Map string labels to integers
    result["label"] = [label2id[l] for l in examples["label"]]
    return result

# Apply preprocessing and remove raw text columns
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

train_ds = processed_dataset["train"]
eval_ds = processed_dataset["test"]

# =========================
# MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# =========================
# METRICS & ARGUMENTS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",  # Updated from evaluation_strategy
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none"
)

# =========================
# TRAINER (FIXED)
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer, # Changed from 'tokenizer' to 'processing_class'
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

print("Starting Training...")
trainer.train()

# =========================
# SAVE MODEL
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Success! Model saved to: {OUTPUT_DIR}")