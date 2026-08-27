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
- **Models**: `distilbert-base-uncased` (as in the paper) and
  `microsoft/codebert-base` as an ablation, each fine-tuned with a
  classification head and class-weighted loss for imbalance
- **Checkpoint selection on macro-F1**, not `eval_loss` — see the note below

## Results

12 epochs, class-weighted cross-entropy, best checkpoint by validation
macro-F1. Test set is the held-out 59-contract split. Both models trained on
CPU (DistilBERT ~7.5 min, CodeBERT ~24 min).

| Model | Accuracy | Macro F1 | Reentrancy F1 | Overflow F1 | Timestamp F1 | Delegatecall F1 |
|---|---|---|---|---|---|---|
| Baseline (TF-IDF+XGBoost) | 0.83 | 0.80 | 0.92 | 0.44 | 0.88 | 0.94 |
| DistilBERT (12 epochs) | 0.83 | 0.81 | **0.96** | 0.44 | 0.85 | **1.00** |
| CodeBERT (12 epochs) | **0.85** | **0.85** | **0.96** | **0.57** | 0.86 | **1.00** |
| Paper (DistilBERT, full IR-Fuzz) | ~90%+ | | ~0.96 | | | |

Both fine-tuned models match the paper's headline Reentrancy F1 (0.96) and are
perfect on Delegatecall. DistilBERT only *ties* the classical baseline on
overall accuracy, though — on 387 contracts TF-IDF is a genuinely strong
competitor, and DistilBERT's edge is in *which* classes it gets right rather
than the aggregate. Swapping in CodeBERT is what actually beats the baseline
outright, on every aggregate measure.

### Integer Overflow: two things were wrong, and only one was a bug

Overflow is the hardest class, and it took two separate fixes to get it from
0.00 to 0.57.

**First, a checkpoint-selection bug.** An early run scored 0.00 — the model
never predicted the class at all. This was not undertraining: with
`metric_for_best_model="eval_loss"`, the majority class dominates validation
loss, so the "best" checkpoint was one that had abandoned Overflow entirely,
and raising 5 → 12 epochs changed nothing because `load_best_model_at_end`
kept reverting to it. Selecting on macro-F1 weights all four classes equally
and recovered the class to 0.44.

**Second, the encoder itself.** At that point both DistilBERT and the TF-IDF
baseline independently sat at exactly 0.44, which looked like a dataset
ceiling from 80 examples. It wasn't — or not entirely. `distilbert-base-uncased`
is pretrained on English prose and *lowercases* its input, so `SafeMath`,
`msg.sender`, and `_transfer` lose casing that carries real meaning in
Solidity. Swapping in `microsoft/codebert-base`, which is pretrained on code
and case-sensitive, lifted Overflow to **0.57** (recall 0.33 → 0.50) without
any other change. The lesson worth keeping: "two different model families
agree on a number" is weaker evidence of a data ceiling than it looks, when
both consume the text the same lossy way.

Overflow is still the weakest class, and the residual error is consistent —
it leaks into Timestamp Dependency, where arithmetic and time-based guards
co-occur (see [`results/confusion_matrix_codebert.png`](results/confusion_matrix_codebert.png)).

## Usage

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt

python src/data_prep.py
python src/baseline.py
python src/train_distilbert.py
python src/evaluate.py

# CodeBERT ablation (same script, different encoder)
python src/train_distilbert.py --model-name microsoft/codebert-base \
    --out-dir models/codebert-vuln
python src/evaluate.py --model-dir models/codebert-vuln --tag codebert

# classify a single snippet
python src/inference.py --file path/to/Contract.sol

# smoke test the trained model
python tests/test_inference.py
```

## Repo layout

```
src/
  data_prep.py        # load, clean, stratified split
  baseline.py         # TF-IDF + XGBoost baseline
  train_distilbert.py # fine-tune an encoder (--model-name to swap it out)
  evaluate.py         # per-class P/R/F1 + confusion matrix
  inference.py        # CLI: classify a Solidity snippet
notebooks/            # exploration
results/              # *_metrics.json, confusion_matrix*.png
tests/                # smoke tests
```

## License

MIT

## About this project

A four-way Solidity vulnerability classifier that flags smart-contract code as
Reentrancy, Integer Overflow, Timestamp Dependency, or Dangerous Delegatecall,
reproducing the approach of an IEEE CCWC 2025 paper on a smaller public
dataset. Built with PyTorch and Hugging Face Transformers, fine-tuning
`distilbert-base-uncased` and `microsoft/codebert-base` with class-weighted
loss against a TF-IDF + XGBoost baseline in scikit-learn/XGBoost, with a CLI
for single-snippet inference. Both models reach **0.96 F1 on Reentrancy** —
matching the paper's headline number — and 1.00 on Delegatecall, with CodeBERT
at 0.85 accuracy / 0.85 macro-F1 on a 387-contract set roughly a sixth the size
of the paper's. The most instructive part was debugging a minority class that
scored 0.00 F1: it turned out to be a checkpoint-selection artifact rather than
undertraining (validation loss is majority-dominated, so selecting on macro-F1
recovered the class), and the residue I had written off as a data-size ceiling
was partly the encoder — an uncased, prose-pretrained tokenizer discards
casing that matters in Solidity, and moving to a code-pretrained model lifted
that class from 0.44 to 0.57.
