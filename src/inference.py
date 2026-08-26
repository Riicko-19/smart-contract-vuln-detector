"""CLI: classify a Solidity snippet's vulnerability type.

Usage:
    python src/inference.py --file path/to/Contract.sol
    python src/inference.py --code "function withdraw() public { ... }"
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-vuln"


def classify(code: str):
    classes = json.loads((MODEL_DIR / "label_classes.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(code, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a .sol file")
    group.add_argument("--code", type=str, help="Raw Solidity snippet")
    args = parser.parse_args()

    code = Path(args.file).read_text() if args.file else args.code
    label, confidence = classify(code)
    print(f"Predicted: {label} (confidence {confidence:.2%})")


if __name__ == "__main__":
    main()
