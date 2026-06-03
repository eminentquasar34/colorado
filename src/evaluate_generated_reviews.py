#!/usr/bin/env python3
"""Score generated reviews with style, plot relevance, and BERTScore metrics."""

from __future__ import annotations

import argparse
import __main__
import json
import re
from pathlib import Path

import joblib
import pandas as pd


PLOT_RE = re.compile(r"Wikipedia plot:\n(?P<plot>.*?)\n\nReview:", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated reviews.")
    parser.add_argument("--generations-jsonl", type=Path, default=Path("outputs/generated_reviews.jsonl"))
    parser.add_argument(
        "--style-model",
        type=Path,
        default=Path("models/ebert_style_classifier/tfidf_logreg.joblib"),
    )
    parser.add_argument(
        "--relevance-model",
        type=Path,
        default=Path("models/plot_relevance_classifier_improved/tfidf_features_logreg.joblib"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/generated_review_scores.csv"))
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip semantic similarity scoring against reference reviews.",
    )
    parser.add_argument(
        "--bertscore-model",
        default="distilbert-base-uncased",
        help="Hugging Face model used by BERTScore.",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=4)
    parser.add_argument(
        "--copy-ngram",
        type=int,
        default=4,
        help="N-gram size for plot copy-rate overlap.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def extract_plot(prompt: str) -> str:
    match = PLOT_RE.search(prompt)
    return clean(match.group("plot")) if match else ""


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(items) < n:
        return []
    return [tuple(items[index : index + n]) for index in range(len(items) - n + 1)]


def copy_rate(source_text: str, generated_text: str, n: int) -> float:
    source_ngrams = set(ngrams(tokens(source_text), n))
    generated_ngrams = ngrams(tokens(generated_text), n)
    if not generated_ngrams:
        return 0.0
    return sum(1 for item in generated_ngrams if item in source_ngrams) / len(generated_ngrams)


def relevance_probability(model: object, plot: str, review: str, title: str) -> float:
    text = f"Plot:\n{plot}\n\nReview:\n{review}"
    try:
        return float(model.predict_proba([text])[0, 1])
    except Exception:
        pair_df = pd.DataFrame(
            [
                {
                    "text": text,
                    "plot_text": plot,
                    "review_text": review,
                    "title": title,
                }
            ]
        )
        return float(model.predict_proba(pair_df)[0, 1])


def main() -> None:
    args = parse_args()
    style_model = joblib.load(args.style_model)["model"]
    try:
        from train_plot_relevance_classifier import PlotRelevanceModel

        __main__.PlotRelevanceModel = PlotRelevanceModel
    except Exception:
        pass
    relevance_model = joblib.load(args.relevance_model)["model"]

    rows = []
    with args.generations_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            generated = clean(record["generated_review"])
            prompt = record["prompt"]
            plot = extract_plot(prompt)
            metadata = record.get("metadata", {})
            reference = clean(record.get("reference_review", ""))
            title = metadata.get("title", "")
            rows.append(
                {
                    "title": title,
                    "movie_year": metadata.get("movie_year", ""),
                    "url": metadata.get("url", ""),
                    "generated_review": generated,
                    "reference_review": reference,
                    "generated_chars": len(generated),
                    "reference_chars": len(reference),
                    f"plot_{args.copy_ngram}gram_copy_rate": copy_rate(
                        plot,
                        generated,
                        args.copy_ngram,
                    ),
                    "style_ebert_probability": style_model.predict_proba([generated])[0, 1],
                    "plot_match_probability": relevance_probability(
                        relevance_model,
                        plot=plot,
                        review=generated,
                        title=title,
                    ),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) and not args.skip_bertscore:
        from bert_score import score

        precision, recall, f1 = score(
            cands=df["generated_review"].tolist(),
            refs=df["reference_review"].tolist(),
            model_type=args.bertscore_model,
            lang="en",
            batch_size=args.bertscore_batch_size,
            verbose=True,
        )
        df["bertscore_precision"] = precision.tolist()
        df["bertscore_recall"] = recall.tolist()
        df["bertscore_f1"] = f1.tolist()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote scores: {args.output_csv}")
    if len(df):
        summary = {
            "style": df["style_ebert_probability"].mean(),
            "plot_match": df["plot_match_probability"].mean(),
            f"plot_{args.copy_ngram}gram_copy": df[f"plot_{args.copy_ngram}gram_copy_rate"].mean(),
            "chars": df["generated_chars"].mean(),
        }
        if "bertscore_f1" in df.columns:
            summary["bertscore_precision"] = df["bertscore_precision"].mean()
            summary["bertscore_recall"] = df["bertscore_recall"].mean()
            summary["bertscore_f1"] = df["bertscore_f1"].mean()
        print(
            "Mean scores: "
            + ", ".join(
                f"{key}={value:.3f}" if key != "chars" else f"{key}={value:.1f}"
                for key, value in summary.items()
            )
        )


if __name__ == "__main__":
    main()
