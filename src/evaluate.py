"""Evaluate a fine-tuned model on the held-out test set.

Defaults to the DistilBERT run; point --model-dir/--tag at another run to score
an ablation, e.g. --model-dir models/codebert-vuln --tag codebert
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def evaluate_model(model_dir: Path):
    """Score a saved model on the held-out test split.

    Returns (report_dict, y_true, preds, classes). Shared with seed_study.py so
    both paths score models identically.
    """
    classes = json.loads((model_dir / "label_classes.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    y_true = [classes.index(l) for l in test_df["label"]]

    preds = []
    with torch.no_grad():
        for text in test_df["code_snippet"]:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds.append(int(logits.argmax(dim=-1)))

    report = classification_report(y_true, preds, target_names=classes, output_dict=True)
    return report, y_true, preds, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/distilbert-vuln")
    parser.add_argument("--tag", default="distilbert", help="prefix for the results files")
    args = parser.parse_args()

    model_dir = ROOT / args.model_dir
    report, y_true, preds, classes = evaluate_model(model_dir)
    print(classification_report(y_true, preds, target_names=classes))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{args.tag}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    cm_path = RESULTS_DIR / ("confusion_matrix.png" if args.tag == "distilbert"
                             else f"confusion_matrix_{args.tag}.png")
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(cm_path)
    print("Saved confusion matrix to", cm_path)


if __name__ == "__main__":
    main()
