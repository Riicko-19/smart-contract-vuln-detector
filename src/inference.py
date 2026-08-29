"""CLI: classify a Solidity snippet's vulnerability type.

Usage:
    python src/inference.py --file path/to/Contract.sol
    python src/inference.py --code "function withdraw() public { ... }"
    python src/inference.py --file Contract.sol --explain
"""
import argparse
import inspect
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explain import VULN_INFO, find_markers, marker_summary, occlusion_attribution

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-vuln"
MAX_LEN = 512


def _load(model_dir: Path):
    classes = json.loads((model_dir / "label_classes.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return classes, tokenizer, model


def _probs(texts: List[str], tokenizer, model) -> List[List[float]]:
    # DistilBERT's tokenizer emits token_type_ids that its model does not accept,
    # so keep only the keys this model's forward() actually takes.
    accepted = set(inspect.signature(model.forward).parameters)
    out = []
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(t, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items() if k in accepted}
            logits = model(**inputs).logits
            out.append(torch.softmax(logits, dim=-1)[0].tolist())
    return out


def classify(code: str, model_dir: Path = MODEL_DIR):
    """Backwards-compatible: returns (label, confidence)."""
    classes, tokenizer, model = _load(model_dir)
    p = _probs([code], tokenizer, model)[0]
    idx = max(range(len(p)), key=p.__getitem__)
    return classes[idx], float(p[idx])


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a .sol file")
    group.add_argument("--code", type=str, help="Raw Solidity snippet")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--explain", action="store_true",
                        help="Also show markers, per-line attribution and remediation")
    args = parser.parse_args()

    code = Path(args.file).read_text(encoding="utf-8", errors="ignore") if args.file else args.code
    model_dir = Path(args.model_dir)
    classes, tokenizer, model = _load(model_dir)

    probs = _probs([code], tokenizer, model)[0]
    order = sorted(range(len(classes)), key=lambda i: probs[i], reverse=True)
    top = order[0]

    print(f"\nPredicted: {classes[top]} ({probs[top]:.1%})\n")
    print("All classes:")
    for i in order:
        bar = "#" * int(round(probs[i] * 30))
        print(f"  {classes[i]:<24} {probs[i]:6.1%}  {bar}")

    if probs[order[0]] - probs[order[1]] < 0.15:
        print(f"\n  Note: top two classes are close ({classes[order[0]]} vs "
              f"{classes[order[1]]}) -- treat this prediction as uncertain.")

    if not args.explain:
        print("\n(run with --explain for markers, per-line attribution and remediation)")
        return

    info = VULN_INFO.get(classes[top])
    if info:
        print(f"\n--- What {classes[top]} means ---")
        print(f"  {info['summary']}")
        print(f"\n  Why it matters: {info['why']}")
        print(f"\n  Typical fix:    {info['fix']}")

    print("\n--- Syntactic markers ---")
    hits = find_markers(code)
    if hits:
        for h in hits:
            print(f"  line {h['line']:>3}  {h['marker']:<32} -> {h['class']}")
    else:
        print("  none found")
    print(f"\n  {marker_summary(code)}")

    print("\n--- Lines the model leaned on ---")
    print("  (drop in predicted-class probability when the line is removed)")
    scored = occlusion_attribution(code, lambda ts: _probs(ts, tokenizer, model), top)
    for line_no, text, imp in scored[:8]:
        sign = "+" if imp >= 0 else "-"
        print(f"  {sign}{abs(imp):5.3f}  line {line_no:>3}  {text[:88]}")
    if not scored:
        print("  (nothing to attribute)")
    print()


if __name__ == "__main__":
    main()
