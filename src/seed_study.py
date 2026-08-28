"""Train each encoder over several seeds and report mean +/- std on the test set.

The test split is only 59 contracts (11 of them Reentrancy), so a single flipped
prediction moves a per-class F1 by roughly 0.09. Single-run numbers are therefore
not trustworthy on their own -- this reruns training across seeds so the README
can quote a mean and a spread instead.

    python src/seed_study.py --seeds 42 43 44 45 46
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

MODELS = [
    ("distilbert", "distilbert-base-uncased"),
    ("codebert", "microsoft/codebert-base"),
]
CLASSES = ["Reentrancy", "Integer Overflow", "Timestamp Dependency", "Dangerous Delegatecall"]


def run_seed(model_name: str, seed: int, out_dir: Path) -> dict:
    """Train one model at one seed, then score it. Returns a flat metrics dict."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "train_distilbert.py"),
         "--model-name", model_name,
         "--out-dir", str(out_dir.relative_to(ROOT)).replace("\\", "/"),
         "--seed", str(seed)],
        check=True, cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    report, _, _, _ = evaluate_model(out_dir)
    row = {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}
    for c in CLASSES:
        row[c] = report[c]["f1-score"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    out = {}
    tmp_dir = ROOT / "models" / "_seed_tmp"
    for tag, model_name in MODELS:
        rows = []
        for seed in args.seeds:
            print(f"[{tag}] seed {seed} ...", flush=True)
            rows.append(run_seed(model_name, seed, tmp_dir))
        keys = rows[0].keys()
        out[tag] = {
            "seeds": args.seeds,
            "runs": rows,
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "std": {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys},
        }
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "seed_study.json", "w") as f:
        json.dump(out, f, indent=2)

    hdr = ["accuracy", "macro_f1"] + CLASSES
    print(f"\n{len(args.seeds)} seeds: {args.seeds}\n")
    print(f"{'model':<12}" + "".join(f"{h[:18]:>20}" for h in hdr))
    for tag, _ in MODELS:
        cells = "".join(f"{out[tag]['mean'][h]:.3f} +/- {out[tag]['std'][h]:.3f}".rjust(20) for h in hdr)
        print(f"{tag:<12}{cells}")
    print("\nSaved", RESULTS_DIR / "seed_study.json")


if __name__ == "__main__":
    main()
