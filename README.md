# smart-contract-vuln-detector

Fine-tuned DistilBERT (+ classical baseline) that classifies Solidity code
snippets into one of four vulnerability types: **Reentrancy**, **Integer
Overflow**, **Timestamp Dependency**, **Dangerous Delegatecall**.

Reproduces the approach from Hossain, Altarawneh & Roberts, ["Leveraging
LLMs and ML for Smart Contract Vulnerability Detection"](https://arxiv.org/abs/2501.02229),
IEEE CCWC 2025.

## Dataset

Labeled Solidity source from the [Messi-Q/Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset)
"Resource 2" release (the dataset lineage IR-Fuzz and the CCWC 2025 paper build
on — the paper's own IR-Fuzz release ships without redistribution rights, so
this is the direct, citable fallback the plan called for). Each vulnerability
type ships as its own binary-labeled folder; positively-labeled contracts from
each are merged into one 4-way multi-class set:

| Class | Count |
|---|---|
| Timestamp Dependency | 174 |
| Integer Overflow | 80 |
| Reentrancy | 71 |
| Dangerous Delegatecall | 62 |

Smaller than the paper's 2,217-contract IR-Fuzz split (their exact curated
release isn't publicly redistributable), so treat results here as directional,
not a reproduction. Not committed to git — fetch it yourself:

```bash
pip install gdown
mkdir -p data/raw
gdown "https://drive.google.com/uc?id=1UhHHevE9iDmvSB_k_lhyI58KAj7hnB1o" -O data/raw/resource2.zip
python -c "import zipfile; zipfile.ZipFile('data/raw/resource2.zip').extractall('data/raw/resource2')"
python src/data_prep.py
```

## Method

- **Baseline**: TF-IDF + XGBoost / Logistic Regression
- **Main model**: `distilbert-base-uncased` fine-tuned with a classification
  head, class-weighted loss for imbalance (Delegatecall is 16% of the data)
- **Checkpoint selection on macro-F1**, not `eval_loss` — see the note below

## Results

12 epochs, `distilbert-base-uncased`, class-weighted cross-entropy, best
checkpoint by validation macro-F1 (epoch 6). Trained on CPU in ~7.5 minutes —
the dataset is small enough that a GPU isn't the bottleneck. Test set is the
held-out 59-contract split.

| Model | Accuracy | Macro F1 | Reentrancy F1 | Overflow F1 | Timestamp F1 | Delegatecall F1 |
|---|---|---|---|---|---|---|
| Baseline (TF-IDF+XGBoost) | 0.83 | 0.80 | 0.92 | 0.44 | 0.88 | 0.94 |
| DistilBERT (12 epochs) | 0.83 | 0.81 | **0.96** | 0.44 | 0.85 | **1.00** |
| Paper (DistilBERT, full IR-Fuzz) | ~90%+ | | ~0.96 | | | |

DistilBERT matches the paper's headline Reentrancy F1 (0.96) and is perfect on
Delegatecall, but only ties the classical baseline on overall accuracy — on
387 contracts, TF-IDF is a genuinely strong competitor, and the honest headline
is that DistilBERT's edge here is in *which* classes it gets right, not the
aggregate.

**Integer Overflow is the weak class (F1 0.44), and the cause is data, not
tuning.** An earlier run scored 0.00 on it — the model never predicted the
class at all. That turned out to be a checkpoint-selection artifact rather
than undertraining: with `metric_for_best_model="eval_loss"`, the majority
class dominates the loss, so the "best" checkpoint was one that had given up
on Overflow entirely, and raising 5 → 12 epochs changed nothing because
`load_best_model_at_end` kept reverting to it. Selecting on macro-F1 instead
weights all four classes equally and recovers the class. What's left is a real
ceiling: 7 of 12 Overflow contracts still leak into Timestamp Dependency (see
[`results/confusion_matrix.png`](results/confusion_matrix.png)), and the
baseline independently lands on the same 0.44 — unsurprising with 80 total
examples, where arithmetic patterns co-occur with timestamp logic.

## Usage

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt

python src/data_prep.py
python src/baseline.py
python src/train_distilbert.py
python src/evaluate.py

# classify a single snippet
python src/inference.py --file path/to/Contract.sol

# smoke test the trained model
python tests/test_inference.py
```

## Repo layout

```
src/
  data_prep.py       # load, clean, stratified split
  baseline.py         # TF-IDF + XGBoost baseline
  train_distilbert.py # fine-tune distilbert-base-uncased
  evaluate.py          # per-class P/R/F1 + confusion matrix
  inference.py         # CLI: classify a Solidity snippet
notebooks/              # exploration
results/                # metrics.json, confusion_matrix.png
tests/                   # smoke tests
```

## License

MIT

## About this project

A four-way Solidity vulnerability classifier that flags smart-contract code as
Reentrancy, Integer Overflow, Timestamp Dependency, or Dangerous Delegatecall,
reproducing the approach of an IEEE CCWC 2025 paper on a smaller public
dataset. Built with PyTorch and Hugging Face Transformers, fine-tuning
`distilbert-base-uncased` with class-weighted loss against a TF-IDF + XGBoost
baseline in scikit-learn/XGBoost, with a CLI for single-snippet inference. The
fine-tuned model reaches **0.96 F1 on Reentrancy** — matching the paper's
headline number — and 1.00 on Delegatecall, at 0.83 overall accuracy on a
387-contract set roughly a sixth the size of the paper's. The most interesting
result was a negative one: a minority class that scored 0.00 F1 turned out to
be a checkpoint-selection artifact, not undertraining — switching model
selection from validation loss to macro-F1 recovered it, and the remaining
error is a genuine data-size ceiling rather than a tuning problem.
