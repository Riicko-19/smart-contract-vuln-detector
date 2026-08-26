"""TF-IDF + XGBoost baseline for sanity-checking the DistilBERT model."""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    le = LabelEncoder().fit(train_df["label"])
    y_train, y_test = le.transform(train_df["label"]), le.transform(test_df["label"])

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(pd.concat([train_df, val_df])["code_snippet"])
    y_train_full = le.transform(pd.concat([train_df, val_df])["label"])
    X_test = vec.transform(test_df["code_snippet"])

    clf = XGBClassifier(n_estimators=200, max_depth=6, eval_metric="mlogloss")
    clf.fit(X_train, y_train_full)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)
    acc = accuracy_score(y_test, preds)
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("Accuracy:", acc)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)


if __name__ == "__main__":
    main()
