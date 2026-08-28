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
macro-F1. Test set is the held-out 59-contract split.

**Numbers are mean ± std over 5 seeds** (42-46), not a single run — see
[`results/seed_study.json`](results/seed_study.json) and the note below on why
that matters here.

| Model | Accuracy | Macro F1 | Reentrancy F1 | Overflow F1 | Timestamp F1 | Delegatecall F1 |
|---|---|---|---|---|---|---|
| Baseline (TF-IDF+XGBoost)¹ | 0.83 | 0.80 | 0.92 | 0.44 | 0.88 | 0.94 |
| DistilBERT | 0.834 ±0.071 | 0.826 ±0.061 | **0.949** ±0.018 | 0.508 ±0.168 | 0.848 ±0.083 | **1.000** ±0.000 |
| CodeBERT | **0.868** ±0.037 | **0.858** ±0.054 | 0.926 ±0.048 | **0.624** ±0.170 | **0.892** ±0.023 | 0.988 ±0.026 |
| Paper (DistilBERT, full IR-Fuzz) | ~90%+ | | ~0.96 | | | |

¹ Baseline is deterministic, so it has no spread.

DistilBERT's Reentrancy F1 of **0.949 ±0.018** is the one result that lines up
cleanly with the paper's ~0.96 headline, and it's the most stable number in the
table. Delegatecall is saturated at 1.000 across every seed. Both fine-tuned
models are at best level with the classical baseline on aggregate accuracy —
on 387 contracts, TF-IDF + XGBoost remains a genuinely strong competitor.

### The honest read on CodeBERT vs DistilBERT: not proven

CodeBERT has the better mean on accuracy (+0.034), macro-F1 (+0.032), Integer
Overflow (+0.116) and Timestamp (+0.044). None of it is statistically
significant at 5 seeds. Paired per-seed comparison:

| Metric | Δ (CodeBERT − DistilBERT) | Seeds won | p (paired t) |
|---|---|---|---|
| Accuracy | +0.034 | 3/5 | 0.41 |
| Macro F1 | +0.032 | 3/5 | 0.47 |
| Integer Overflow | +0.116 | 3/5 | 0.38 |
| Reentrancy | −0.022 | 0/2 | 0.20 |
| Delegatecall | −0.012 | 0/1 | 0.37 |

CodeBERT trends better on the aggregate and on Overflow; DistilBERT is
*better* on the two classes it already handles well. With a 59-contract test
set, this dataset cannot separate the two models.

### Integer Overflow: one real bug, and one lesson about small test sets

**The real bug — checkpoint selection.** An early run scored 0.00 F1 on Integer
Overflow: the model never predicted the class at all. This was not
undertraining. With `metric_for_best_model="eval_loss"`, the majority class
dominates validation loss, so the "best" checkpoint was one that had abandoned
Overflow entirely, and raising 5 → 12 epochs changed nothing because
`load_best_model_at_end` kept reverting to it. Selecting on macro-F1 weights all
four classes equally and fixed it. That result is reproducible and not noise.

**The lesson — single runs on 59 samples say very little.** Integer Overflow
swings by ±0.17 across seeds (DistilBERT: 0.26 → 0.73; CodeBERT: 0.35 → 0.78).
Earlier iterations of this README drew confident conclusions from single runs —
that 0.44 was a "data ceiling" because two model families agreed on it, then
that CodeBERT's 0.57 proved the ceiling was really the encoder. Both readings
were artifacts of run-to-run variance far larger than the effects being
described. With 11 Reentrancy and 12 Overflow contracts in the test split, one
flipped prediction moves a per-class F1 by ~0.09, which is why everything here
is reported as mean ± std.

Overflow remains the weakest class under any measurement, and its errors leak
into Timestamp Dependency, where arithmetic and time-based guards co-occur (see
[`results/confusion_matrix_codebert.png`](results/confusion_matrix_codebert.png),
a representative single run).

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

# multi-seed study -> results/seed_study.json (mean +/- std, both models)
python src/seed_study.py --seeds 42 43 44 45 46

# classify a single snippet
python src/inference.py --file path/to/Contract.sol

# smoke test the trained model
python tests/test_inference.py
```

Training runs on CPU or GPU unchanged — `fp16` switches on automatically when
CUDA/ROCm is available. Reference timings for the full 12-epoch run: DistilBERT
454s CPU vs 64s GPU, CodeBERT 1430s CPU vs 70s GPU (Ryzen CPU / Radeon RX 9060
XT, ROCm 10). CPU and GPU runs do not produce bit-identical results, which is
part of why results are reported as mean ± std over seeds.

## Repo layout

```
src/
  data_prep.py        # load, clean, stratified split
  baseline.py         # TF-IDF + XGBoost baseline
  train_distilbert.py # fine-tune an encoder (--model-name to swap it out)
  evaluate.py         # per-class P/R/F1 + confusion matrix
  seed_study.py       # retrain over N seeds -> mean +/- std
  inference.py        # CLI: classify a Solidity snippet
notebooks/            # exploration
results/              # *_metrics.json, seed_study.json, confusion_matrix*.png
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
for single-snippet inference. Over 5 seeds, DistilBERT reaches **0.949 ±0.018
F1 on Reentrancy** — in line with the paper's ~0.96 headline — and a perfect
1.000 on Delegatecall, at 0.83 accuracy on a 387-contract set roughly a sixth
the size of the paper's. Two findings drove most of the work: a minority class
scoring 0.00 F1 turned out to be a checkpoint-selection artifact rather than
undertraining (validation loss is majority-dominated, so selecting on macro-F1
recovered the class), and a CodeBERT ablation that looked like a clear win on
single runs proved statistically indistinguishable once measured across seeds —
per-class F1 varies by ±0.17 on a 59-contract test split, wider than any effect
being claimed. The repo reports mean ± std for that reason.
