#!/usr/bin/env python3
"""Train a baseline classifier for Roger Ebert versus non-Ebert reviews."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


EDITOR_NOTE_RE = re.compile(r"^\s*\[Editor[’']s note:.*?\]\s*", re.IGNORECASE | re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + logistic regression Ebert style classifier."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/modeling/classifier_train.csv"),
        help="Prepared classifier training split. Created by src/create_modeling_splits.py.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/modeling/classifier_val.csv"),
        help="Prepared classifier validation split. Created by src/create_modeling_splits.py.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/modeling/classifier_test.csv"),
        help="Prepared classifier test split. Created by src/create_modeling_splits.py.",
    )
    parser.add_argument(
        "--ebert-csv",
        type=Path,
        default=Path("data/processed/ebert_reviews_full.csv"),
        help="Fallback CSV containing Roger Ebert reviews if prepared splits do not exist.",
    )
    parser.add_argument(
        "--non-ebert-csv",
        type=Path,
        default=Path("data/raw/non_ebert_reviews/non_ebert_reviews.csv"),
        help="Fallback CSV containing non-Ebert reviews if prepared splits do not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/ebert_style_classifier"),
        help="Directory for model and metric artifacts.",
    )
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument(
        "--keep-editor-notes",
        action="store_true",
        help="Keep leading RogerEbert.com editor notes in review text.",
    )
    return parser.parse_args()


def clean_text(text: str, strip_editor_notes: bool) -> str:
    text = str(text or "")
    if strip_editor_notes:
        text = EDITOR_NOTE_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def load_labeled_reviews(
    ebert_csv: Path,
    non_ebert_csv: Path,
    min_chars: int,
    strip_editor_notes: bool,
) -> pd.DataFrame:
    frames = []
    for path, label, source in [
        (ebert_csv, 1, "ebert"),
        (non_ebert_csv, 0, "non_ebert"),
    ]:
        df = pd.read_csv(path)
        required = {"review_text", "title", "author", "published_date", "rating", "url"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df = df.copy()
        df["review_text"] = df["review_text"].fillna("").map(
            lambda value: clean_text(value, strip_editor_notes=strip_editor_notes)
        )
        df["text_length"] = df["review_text"].str.len()
        df = df[df["text_length"] >= min_chars]
        df = df.drop_duplicates(subset=["url"])
        df["label"] = label
        df["source"] = source
        frames.append(
            df[
                [
                    "label",
                    "source",
                    "review_text",
                    "text_length",
                    "title",
                    "author",
                    "published_date",
                    "rating",
                    "url",
                ]
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return combined


def make_splits(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("--val-size and --test-size must be positive and sum to less than 1.")

    train_df, holdout_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        stratify=holdout_df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_prepared_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"label", "source", "review_text", "title", "author", "published_date", "rating", "url"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def build_model(max_features: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["label"].to_numpy()
    y_pred = model.predict(df["review_text"])
    y_prob = model.predict_proba(df["review_text"])[:, 1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["non_ebert", "ebert"],
            output_dict=True,
        ),
    }


def top_features(model: Pipeline, limit: int = 40) -> dict[str, list[dict[str, float | str]]]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    ebert_idx = coefficients.argsort()[-limit:][::-1]
    non_ebert_idx = coefficients.argsort()[:limit]
    return {
        "ebert": [
            {"feature": names[index], "coefficient": float(coefficients[index])}
            for index in ebert_idx
        ],
        "non_ebert": [
            {"feature": names[index], "coefficient": float(coefficients[index])}
            for index in non_ebert_idx
        ],
    }


def write_predictions(model: Pipeline, df: pd.DataFrame, path: Path) -> None:
    output = df[
        ["label", "source", "title", "author", "published_date", "rating", "url", "text_length"]
    ].copy()
    output["predicted_label"] = model.predict(df["review_text"])
    output["ebert_probability"] = model.predict_proba(df["review_text"])[:, 1]
    output.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv.exists() and args.val_csv.exists() and args.test_csv.exists():
        train_df = load_prepared_split(args.train_csv)
        val_df = load_prepared_split(args.val_csv)
        test_df = load_prepared_split(args.test_csv)
        reviews = pd.concat([train_df, val_df, test_df], ignore_index=True)
        split_source = "prepared"
    else:
        strip_editor_notes = not args.keep_editor_notes
        reviews = load_labeled_reviews(
            ebert_csv=args.ebert_csv,
            non_ebert_csv=args.non_ebert_csv,
            min_chars=args.min_chars,
            strip_editor_notes=strip_editor_notes,
        )
        train_df, val_df, test_df = make_splits(
            reviews,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        split_source = "generated_by_training_script"

    model = build_model(max_features=args.max_features)
    model.fit(train_df["review_text"], train_df["label"])

    metrics = {
        "dataset": {
            "total_rows": int(len(reviews)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "label_counts": reviews["source"].value_counts().to_dict(),
            "min_chars": args.min_chars,
            "split_source": split_source,
            "train_csv": str(args.train_csv) if args.train_csv.exists() else None,
            "val_csv": str(args.val_csv) if args.val_csv.exists() else None,
            "test_csv": str(args.test_csv) if args.test_csv.exists() else None,
            "strip_editor_notes": None if split_source == "prepared" else strip_editor_notes,
        },
        "validation": evaluate(model, val_df),
        "test": evaluate(model, test_df),
        "top_features": top_features(model),
    }

    metrics_path = args.output_dir / "metrics.json"
    model_path = args.output_dir / "tfidf_logreg.joblib"
    val_predictions_path = args.output_dir / "validation_predictions.csv"
    test_predictions_path = args.output_dir / "test_predictions.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": model,
            "label_mapping": {"non_ebert": 0, "ebert": 1},
            "metadata": metrics["dataset"],
        },
        model_path,
    )
    write_predictions(model, val_df, val_predictions_path)
    write_predictions(model, test_df, test_predictions_path)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Validation "
        f"accuracy={metrics['validation']['accuracy']:.3f} "
        f"f1={metrics['validation']['f1']:.3f} "
        f"roc_auc={metrics['validation']['roc_auc']:.3f}"
    )
    print(
        "Test "
        f"accuracy={metrics['test']['accuracy']:.3f} "
        f"f1={metrics['test']['f1']:.3f} "
        f"roc_auc={metrics['test']['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
