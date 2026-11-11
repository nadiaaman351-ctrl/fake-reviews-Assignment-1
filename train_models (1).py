"""
Member 3 - Machine Learning Model
Fake Reviews Detection: Naive Bayes vs Logistic Regression vs SVM

Run:
    python train_models.py --data data/cleaned_reviews.csv
    python train_models.py --data data/cleaned_reviews.csv --tune

Expected input: a CSV with a text column and a label column.
Raw Kaggle file columns: category, rating, label (CG / OR), text_
Phase 1 cleaned file: auto-detects clean_text / text_clean / text_ / text.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
TEXT_CANDIDATES = ["clean_text", "text_clean", "text_", "text"]

# CG = computer-generated (fake), OR = original (real)
LABEL_MAP = {"CG": 1, "OR": 0, "fake": 1, "real": 0, 1: 1, 0: 0}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: Path, text_col: str | None, label_col: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)

    if text_col is None:
        text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
        if text_col is None:
            raise ValueError(f"No text column found. Columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in {list(df.columns)}")

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip().str.len() > 0]
    n_before = len(df)
    df = df.drop_duplicates(subset=text_col)  # identical texts could land in both train and test
    if n_before != len(df):
        print(f"Dropped {n_before - len(df)} duplicate texts")

    y = df[label_col].map(LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"Unmapped labels: {df.loc[y.isna(), label_col].unique()}")

    print(f"Loaded {len(df):,} rows | text='{text_col}' | label='{label_col}'")
    print("Class balance (1 = fake):")
    print(y.value_counts(normalize=True).round(3).to_string(), "\n")
    return df[text_col], y.astype(int)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=200_000,
    )


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """Each model is a Pipeline so TF-IDF is fit on training data only (no leakage)."""
    return {
        "Naive Bayes": (
            Pipeline([("tfidf", make_tfidf()), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "Logistic Regression": (
            Pipeline([
                ("tfidf", make_tfidf()),
                ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.5, 1, 5, 10]},
        ),
        "SVM (LinearSVC)": (
            Pipeline([
                ("tfidf", make_tfidf()),
                ("clf", LinearSVC(random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.1, 0.5, 1, 5]},
        ),
    }


def get_scores(model: Pipeline, X):
    """Continuous score for ROC-AUC (probabilities or margins)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default="label")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--tune", action="store_true", help="3-fold grid search on the training set")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models").mkdir(exist_ok=True)

    X, y = load_data(args.data, args.text_col, args.label_col)

    # Stratified split keeps the 50:50 ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    results, fig_cm = [], plt.subplots(1, 3, figsize=(15, 4.5))
    fig, axes = fig_cm

    for ax, (name, (pipe, grid)) in zip(axes, build_models().items()):
        print(f"=== {name} ===")
        t0 = time.perf_counter()

        if args.tune:
            search = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1)
            search.fit(X_train, y_train)
            model, best = search.best_estimator_, search.best_params_
            print(f"Best params: {best}")
        else:
            model, best = pipe.fit(X_train, y_train), {}

        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = model.predict(X_test)
        pred_time = time.perf_counter() - t1

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, get_scores(model, X_test)),
            "train_time_s": round(train_time, 2),
            "predict_time_s": round(pred_time, 3),
            "best_params": json.dumps(best),
        }
        results.append(row)

        print(classification_report(y_test, pred, target_names=["real (OR)", "fake (CG)"], digits=4))

        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["real", "fake"], cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"{name}\nF1 = {row['f1']:.3f}")

        slug = {"Naive Bayes": "naive_bayes", "Logistic Regression": "logistic_regression"}.get(name, "svm")
        joblib.dump(model, args.out / "models" / f"{slug}.joblib")

    fig.tight_layout()
    fig.savefig(args.out / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ---- comparison table + chart ----
    res = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    res.to_csv(args.out / "model_comparison.csv", index=False)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = res.set_index("model")[metrics].plot(kind="bar", figsize=(10, 5), rot=0, ylim=(0, 1.05))
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out / "model_comparison.png", dpi=150)
    plt.close()

    print("=== Comparison (sorted by F1) ===")
    print(res.drop(columns="best_params").round(4).to_string(index=False))
    print(f"\nBest model: {res.loc[0, 'model']}")
    print(f"Saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
