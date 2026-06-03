#!/usr/bin/env python3
"""Train an improved classifier for plot/review relevance."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TOKEN_RE = re.compile(r"[a-z0-9']+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a plot-review relevance classifier with hard negatives."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/modeling/ebert_generator_train_with_plots.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/modeling/ebert_val_with_plots.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/modeling/ebert_test_with_plots.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/plot_relevance_classifier"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=120_000)
    parser.add_argument(
        "--negative-types",
        default="random,same_decade,same_genre,nearest_tfidf",
        help="Comma-separated negative types to create per positive.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(clean(text).lower())


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(items) < n:
        return []
    return [tuple(items[index : index + n]) for index in range(len(items) - n + 1)]


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def copy_rate(source_text: object, generated_text: object, n: int = 4) -> float:
    source_ngrams = set(ngrams(tokens(source_text), n))
    generated_ngrams = ngrams(tokens(generated_text), n)
    if not generated_ngrams:
        return 0.0
    return sum(1 for item in generated_ngrams if item in source_ngrams) / len(generated_ngrams)


def extract_year(value: object) -> int | None:
    match = YEAR_RE.search(clean(value))
    return int(match.group(0)) if match else None


def decade(value: object) -> int | None:
    year = extract_year(value)
    return (year // 10) * 10 if year else None


def genre_tokens(value: object) -> set[str]:
    text = clean(value).lower()
    return {part for part in re.split(r"[^a-z0-9]+", text) if part and part != "unknown"}


def load_matched(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"title", "review_text", "wiki_plot", "url"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df[df["wiki_plot"].fillna("").astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    df["title_clean"] = df["title"].map(clean)
    df["movie_decade"] = df.get("movie_year", pd.Series([""] * len(df))).map(decade)
    df["genre_set"] = df.get("wiki_genre", pd.Series([""] * len(df))).map(genre_tokens)
    return df


def pair_text(plot: object, review: object) -> str:
    return f"Plot:\n{clean(plot)}\n\nReview:\n{clean(review)}"


def choose_random_wrong(df: pd.DataFrame, row_index: int, rng: np.random.Generator) -> int:
    if len(df) <= 1:
        return row_index
    candidate = int(rng.integers(0, len(df) - 1))
    return candidate if candidate < row_index else candidate + 1


def choose_same_decade_wrong(df: pd.DataFrame, row_index: int, rng: np.random.Generator) -> int:
    current_decade = df.loc[row_index, "movie_decade"]
    candidates = df.index[(df.index != row_index) & (df["movie_decade"].eq(current_decade))].to_numpy()
    if len(candidates) == 0:
        return choose_random_wrong(df, row_index, rng)
    return int(rng.choice(candidates))


def choose_same_genre_wrong(df: pd.DataFrame, row_index: int, rng: np.random.Generator) -> int:
    current_genres = df.loc[row_index, "genre_set"]
    candidates = [
        index
        for index, genres in df["genre_set"].items()
        if index != row_index and bool(current_genres & genres)
    ]
    if not candidates:
        return choose_random_wrong(df, row_index, rng)
    return int(rng.choice(candidates))


def nearest_plot_indices(df: pd.DataFrame) -> list[int]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(df["wiki_plot"].map(clean))
    scores = linear_kernel(matrix, matrix)
    np.fill_diagonal(scores, -1.0)
    return scores.argmax(axis=1).astype(int).tolist()


def make_pairs(
    df: pd.DataFrame,
    random_state: int,
    negative_types: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    print(f"Building {len(df)} positives with negatives: {', '.join(negative_types)}")
    nearest = nearest_plot_indices(df) if "nearest_tfidf" in negative_types else None
    rows = []

    for index, row in df.iterrows():
        query_id = clean(row["url"]) or str(index)
        rows.append(
            {
                "query_id": query_id,
                "text": pair_text(row["wiki_plot"], row["review_text"]),
                "plot_text": clean(row["wiki_plot"]),
                "review_text": clean(row["review_text"]),
                "label": 1,
                "title": clean(row["title"]),
                "url": clean(row["url"]),
                "pair_type": "matched",
            }
        )

        for negative_type in negative_types:
            if negative_type == "random":
                negative_index = choose_random_wrong(df, index, rng)
            elif negative_type == "same_decade":
                negative_index = choose_same_decade_wrong(df, index, rng)
            elif negative_type == "same_genre":
                negative_index = choose_same_genre_wrong(df, index, rng)
            elif negative_type == "nearest_tfidf":
                if nearest is None:
                    raise ValueError("nearest_tfidf requested but nearest indices were not built.")
                negative_index = nearest[index]
            else:
                raise ValueError(f"Unknown negative type: {negative_type}")

            negative_plot = df.loc[negative_index, "wiki_plot"]
            rows.append(
                {
                    "query_id": query_id,
                    "text": pair_text(negative_plot, row["review_text"]),
                    "plot_text": clean(negative_plot),
                    "review_text": clean(row["review_text"]),
                    "label": 0,
                    "title": clean(row["title"]),
                    "url": clean(row["url"]),
                    "pair_type": negative_type,
                }
            )

    return pd.DataFrame(rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def feature_names() -> list[str]:
    return [
        "plot_token_count",
        "review_token_count",
        "length_ratio",
        "token_jaccard",
        "title_in_plot",
        "title_in_review",
        "plot_3gram_copy_rate",
        "plot_4gram_copy_rate",
    ]


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    features = []
    for _, row in df.iterrows():
        plot_tokens = tokens(row["plot_text"])
        review_tokens = tokens(row["review_text"])
        plot_set = set(plot_tokens)
        review_set = set(review_tokens)
        title = clean(row["title"]).lower()
        plot = clean(row["plot_text"]).lower()
        review = clean(row["review_text"]).lower()
        features.append(
            [
                len(plot_tokens),
                len(review_tokens),
                len(review_tokens) / max(1, len(plot_tokens)),
                overlap_ratio(plot_set, review_set),
                1.0 if title and title in plot else 0.0,
                1.0 if title and title in review else 0.0,
                copy_rate(plot, review, n=3),
                copy_rate(plot, review, n=4),
            ]
        )
    return np.asarray(features, dtype=float)


class PlotRelevanceModel:
    def __init__(self, max_features: int, random_state: int) -> None:
        self.text_model = Pipeline(
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
            ]
        )
        self.feature_scaler = StandardScaler()
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="liblinear",
        )

    def _matrix(self, df: pd.DataFrame, fit: bool = False) -> csr_matrix:
        if fit:
            text_matrix = self.text_model.fit_transform(df["text"])
            feature_matrix = self.feature_scaler.fit_transform(numeric_features(df))
        else:
            text_matrix = self.text_model.transform(df["text"])
            feature_matrix = self.feature_scaler.transform(numeric_features(df))
        return hstack([text_matrix, csr_matrix(feature_matrix)])

    def fit(self, df: pd.DataFrame) -> "PlotRelevanceModel":
        matrix = self._matrix(df, fit=True)
        self.classifier.fit(matrix, df["label"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict(self._matrix(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self._matrix(df))


def evaluate(model: PlotRelevanceModel, df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["label"].to_numpy()
    y_pred = model.predict(df)
    y_prob = model.predict_proba(df)[:, 1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["wrong_plot", "matching_plot"],
            output_dict=True,
        ),
        "ranking": ranking_metrics(df.assign(matching_plot_probability=y_prob)),
    }


def ranking_metrics(scored: pd.DataFrame) -> dict[str, float]:
    reciprocal_ranks = []
    top1 = []
    recall_at_3 = []
    for _, group in scored.groupby("query_id"):
        ranked = group.sort_values("matching_plot_probability", ascending=False).reset_index(drop=True)
        positive_positions = ranked.index[ranked["label"].eq(1)].tolist()
        if not positive_positions:
            continue
        rank = positive_positions[0] + 1
        reciprocal_ranks.append(1.0 / rank)
        top1.append(1.0 if rank == 1 else 0.0)
        recall_at_3.append(1.0 if rank <= 3 else 0.0)
    return {
        "top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "recall_at_3": float(np.mean(recall_at_3)) if recall_at_3 else 0.0,
    }


def write_predictions(model: PlotRelevanceModel, df: pd.DataFrame, path: Path) -> None:
    output = df[["query_id", "label", "title", "url", "pair_type"]].copy()
    output["predicted_label"] = model.predict(df)
    output["matching_plot_probability"] = model.predict_proba(df)[:, 1]
    output.to_csv(path, index=False)


def write_pair_data(df: pd.DataFrame, path: Path) -> None:
    columns = ["query_id", "label", "title", "url", "pair_type", "plot_text", "review_text", "text"]
    df[columns].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    negative_types = [item.strip() for item in args.negative_types.split(",") if item.strip()]

    train_df = make_pairs(
        load_matched(args.train_csv),
        random_state=args.random_state,
        negative_types=negative_types,
    )
    print(f"Train pairs: {len(train_df)}")
    val_df = make_pairs(
        load_matched(args.val_csv),
        random_state=args.random_state,
        negative_types=negative_types,
    )
    print(f"Validation pairs: {len(val_df)}")
    test_df = make_pairs(
        load_matched(args.test_csv),
        random_state=args.random_state,
        negative_types=negative_types,
    )
    print(f"Test pairs: {len(test_df)}")

    model = PlotRelevanceModel(max_features=args.max_features, random_state=args.random_state)
    print("Training classifier...")
    model.fit(train_df)
    print("Evaluating...")

    metrics = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_queries": int(train_df["query_id"].nunique()),
            "val_queries": int(val_df["query_id"].nunique()),
            "test_queries": int(test_df["query_id"].nunique()),
            "negatives_per_positive": len(negative_types),
            "negative_types": negative_types,
            "feature_names": feature_names(),
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "test_csv": str(args.test_csv),
        },
        "validation": evaluate(model, val_df),
        "test": evaluate(model, test_df),
    }

    metrics_path = args.output_dir / "metrics.json"
    model_path = args.output_dir / "tfidf_features_logreg.joblib"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": model,
            "label_mapping": {"wrong_plot": 0, "matching_plot": 1},
            "negative_types": negative_types,
            "feature_names": feature_names(),
        },
        model_path,
    )
    write_predictions(model, val_df, args.output_dir / "validation_predictions.csv")
    write_predictions(model, test_df, args.output_dir / "test_predictions.csv")
    write_pair_data(train_df, args.output_dir / "train_pairs.csv")
    write_pair_data(val_df, args.output_dir / "validation_pairs.csv")
    write_pair_data(test_df, args.output_dir / "test_pairs.csv")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Test "
        f"accuracy={metrics['test']['accuracy']:.3f} "
        f"f1={metrics['test']['f1']:.3f} "
        f"roc_auc={metrics['test']['roc_auc']:.3f} "
        f"top1={metrics['test']['ranking']['top1_accuracy']:.3f} "
        f"mrr={metrics['test']['ranking']['mrr']:.3f}"
    )


if __name__ == "__main__":
    main()
