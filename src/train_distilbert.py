"""Fine-tune distilbert-base-uncased for 4-class Solidity vulnerability classification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-vuln"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Macro-F1 weights every class equally, so the minority classes count as much
    as Timestamp Dependency (45% of the data). Selecting the best checkpoint on
    eval_loss instead lets the model win on loss while never predicting Integer
    Overflow at all -- macro-F1 is what we actually care about here.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": float((preds == labels).mean()),
    }


def load_split(name: str, le: LabelEncoder) -> Dataset:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["label_id"] = le.transform(df["label"])
    return Dataset.from_pandas(df[["code_snippet", "label_id"]].rename(columns={"label_id": "labels"}))


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    le = LabelEncoder().fit(train_df["label"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["code_snippet"], truncation=True, max_length=MAX_LEN)

    train_ds = load_split("train", le).map(tokenize, batched=True).remove_columns(["code_snippet"])
    val_ds = load_split("val", le).map(tokenize, batched=True).remove_columns(["code_snippet"])

    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(len(le.classes_)), y=le.transform(train_df["label"])),
        dtype=torch.float,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=12,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    with open(MODEL_DIR / "label_classes.json", "w") as f:
        json.dump(list(le.classes_), f)
    print("Saved model to", MODEL_DIR)


if __name__ == "__main__":
    main()
