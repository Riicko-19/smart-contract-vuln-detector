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
  head, class-weighted loss for imbalance (Delegatecall is 4% of the data)

## Results

Trained locally on CPU (5 epochs) as a first pass — see
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) to retrain on a
GPU with more epochs for the final numbers.

| Model | Accuracy | Reentrancy F1 | Overflow F1 | Timestamp F1 | Delegatecall F1 |
|---|---|---|---|---|---|
| Baseline (TF-IDF+XGBoost) | 0.83 | 0.92 | 0.44 | 0.88 | 0.94 |
| DistilBERT (5 epochs, CPU) | 0.80 | 0.85 | 0.00 | 0.90 | 0.90 |
| Paper (DistilBERT, full IR-Fuzz) | ~90%+ | ~0.96 | | | |

DistilBERT undertrained on Integer Overflow at 5 epochs on this smaller
dataset (56 train examples for that class) — loss was still falling each
epoch, so a longer GPU run (see Colab notebook, 12+ epochs) is expected to
close this gap; that's the immediate next step.

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
