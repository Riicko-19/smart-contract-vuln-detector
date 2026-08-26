"""Build a 4-class (code_snippet, label) DataFrame from the Messi-Q smart-contract
vulnerability dataset (the labeled source cited by IR-Fuzz / the CCWC 2025 paper).

Each vulnerability type ships as its own binary-labeled folder (label 1 = contract
exhibits that vulnerability). We take the positively-labeled contracts from each of
the four folders and merge them into one multi-class dataset, one class per contract.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "resource2" / "dataset_preprocessing_for_vulnerabilities"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# (folder name, label file prefix, class label). Prefer the "_fixed" label files
# where available -- they're a deduplicated/corrected pass over the raw ones.
CLASSES = [
    ("reentrancy", "final_reentrancy_name_fixed.txt", "final_reentrancy_label_fixed.txt", "Reentrancy"),
    ("Integeroverflow", "final_integeroverflow_name_fixed.txt", "final_integeroverflow_label_fixed.txt", "Integer Overflow"),
    ("timestamp", "final_timestamp_name_fixed.txt", "final_timestamp_label_fixed.txt", "Timestamp Dependency"),
    ("delegatecall", "final_delegatecall_name.txt", "final_delegatecall_label.txt", "Dangerous Delegatecall"),
]


def load_class(folder: str, name_file: str, label_file: str, label: str) -> pd.DataFrame:
    base = RAW_DIR / folder
    names = (base / name_file).read_text().split()
    labels = (base / label_file).read_text().split()
    rows = []
    for fname, is_vuln in zip(names, labels):
        if is_vuln != "1":
            continue
        f = base / "sourcecode" / fname
        if not f.exists():
            continue
        code = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not code:
            continue
        rows.append({"code_snippet": code, "label": label})
    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    frames = [load_class(*c) for c in CLASSES]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="code_snippet").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_dataset()
    print("Class distribution:\n", df["label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=args.val_size / (1 - args.test_size),
        stratify=train_df["label"], random_state=args.seed,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
