"""Evaluate the fine-tuned DistilBERT model on the held-out test set."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-vuln"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    classes = json.loads((MODEL_DIR / "label_classes.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    y_true = [classes.index(l) for l in test_df["label"]]

    preds = []
    with torch.no_grad():
        for text in test_df["code_snippet"]:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            logits = model(**inputs).logits
            preds.append(int(logits.argmax(dim=-1)))

    report = classification_report(y_true, preds, target_names=classes, output_dict=True)
    print(classification_report(y_true, preds, target_names=classes))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "distilbert_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    print("Saved confusion matrix to", RESULTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
