#!/usr/bin/env python3
"""Build prompt/completion JSONL files for Ebert-style review generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create generator-ready JSONL from Ebert reviews with matched plots."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/modeling/ebert_generator_train_with_plots.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/modeling/generator_train.jsonl"),
    )
    parser.add_argument(
        "--require-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows without matched wiki_plot text.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_prompt(row: pd.Series) -> str:
    title = clean(row.get("title", ""))
    year = clean(row.get("movie_year", "")) or clean(row.get("wiki_release_year", ""))
    rating = clean(row.get("rating", ""))
    plot = clean(row.get("wiki_plot", ""))

    details = [f"Movie: {title}"]
    if year:
        details.append(f"Year: {year}")
    if rating:
        details.append(f"Roger Ebert rating: {rating} / 4")

    return (
        "Write a film review in the style of Roger Ebert, grounded in the movie plot.\n\n"
        + "\n".join(details)
        + "\n\nWikipedia plot:\n"
        + plot
        + "\n\nReview:"
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"review_text", "title", "wiki_plot"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{args.input_csv} is missing required columns: {missing}")

    if args.require_plot:
        df = df[df["wiki_plot"].fillna("").astype(str).str.strip().ne("")]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            record = {
                "prompt": build_prompt(row),
                "completion": clean(row["review_text"]),
                "metadata": {
                    "title": clean(row.get("title", "")),
                    "movie_year": clean(row.get("movie_year", "")),
                    "rating": clean(row.get("rating", "")),
                    "url": clean(row.get("url", "")),
                    "wiki_page": clean(row.get("wiki_page", "")),
                    "plot_match_status": clean(row.get("plot_match_status", "")),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} generator examples: {args.output_jsonl}")


if __name__ == "__main__":
    main()
