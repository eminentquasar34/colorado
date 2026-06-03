#!/usr/bin/env python3
"""Create reusable modeling splits for generator and classifier experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


EDITOR_NOTE_RE = re.compile(r"^\s*\[Editor[’']s note:.*?\]\s*", re.IGNORECASE | re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
BASE_COLUMNS = [
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
OPTIONAL_METADATA_COLUMNS = [
    "movie_year",
    "index_year",
    "director",
    "genres",
    "cast",
    "runtime_minutes",
    "mpaa_rating",
    "image_url",
    "short_description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed, non-overlapping Ebert/non-Ebert modeling splits."
    )
    parser.add_argument(
        "--ebert-csv",
        type=Path,
        default=Path("data/processed/ebert_reviews_full.csv"),
    )
    parser.add_argument(
        "--non-ebert-csv",
        type=Path,
        default=Path("data/raw/non_ebert_reviews/non_ebert_reviews.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/modeling"),
    )
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument(
        "--generator-size",
        type=float,
        default=0.60,
        help="Fraction of Ebert reviews reserved for generator training.",
    )
    parser.add_argument(
        "--classifier-size",
        type=float,
        default=0.20,
        help="Fraction of Ebert reviews reserved as classifier positive training rows.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Fraction of Ebert reviews reserved for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Fraction of Ebert reviews reserved for locked final evaluation.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Non-Ebert negatives per Ebert positive in classifier splits.",
    )
    parser.add_argument("--random-state", type=int, default=42)
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


def load_reviews(path: Path, label: int, source: str, min_chars: int, strip_editor_notes: bool) -> pd.DataFrame:
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
    output_columns = BASE_COLUMNS + [
        column for column in OPTIONAL_METADATA_COLUMNS if column in df.columns
    ]
    return df[output_columns].reset_index(drop=True)


def validate_split_sizes(generator_size: float, classifier_size: float, val_size: float, test_size: float) -> None:
    sizes = {
        "--generator-size": generator_size,
        "--classifier-size": classifier_size,
        "--val-size": val_size,
        "--test-size": test_size,
    }
    for name, size in sizes.items():
        if size < 0:
            raise ValueError(f"{name} must be non-negative.")

    total = sum(sizes.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "--generator-size, --classifier-size, --val-size, and --test-size must sum to 1.0."
        )


def split_by_fractions(
    df: pd.DataFrame,
    fractions: dict[str, float],
    random_state: int,
) -> dict[str, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    total = len(shuffled)
    names = list(fractions)

    counts: dict[str, int] = {}
    assigned = 0
    for name in names[:-1]:
        count = int(total * fractions[name])
        counts[name] = count
        assigned += count
    counts[names[-1]] = total - assigned

    splits = {}
    start = 0
    for name in names:
        end = start + counts[name]
        splits[name] = shuffled.iloc[start:end].reset_index(drop=True)
        start = end
    return splits


def take_non_ebert_splits(
    non_ebert: pd.DataFrame,
    target_counts: dict[str, int],
    negative_ratio: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    if negative_ratio <= 0:
        raise ValueError("--negative-ratio must be positive.")

    counts = {
        name: int(round(count * negative_ratio))
        for name, count in target_counts.items()
    }
    required = sum(counts.values())
    if required > len(non_ebert):
        raise ValueError(
            f"Need {required} non-Ebert reviews for negative_ratio={negative_ratio}, "
            f"but only {len(non_ebert)} are available."
        )

    shuffled = non_ebert.sample(frac=1.0, random_state=random_state + 1).reset_index(drop=True)
    splits = {}
    start = 0
    for name, count in counts.items():
        end = start + count
        splits[name] = shuffled.iloc[start:end].reset_index(drop=True)
        start = end
    splits["unused"] = shuffled.iloc[start:].reset_index(drop=True)
    return splits


def mix_classifier_split(
    ebert_df: pd.DataFrame,
    non_ebert_df: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    return (
        pd.concat([ebert_df, non_ebert_df], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):5d} rows: {path}")


def main() -> None:
    args = parse_args()
    validate_split_sizes(args.generator_size, args.classifier_size, args.val_size, args.test_size)
    strip_editor_notes = not args.keep_editor_notes

    ebert = load_reviews(
        args.ebert_csv,
        label=1,
        source="ebert",
        min_chars=args.min_chars,
        strip_editor_notes=strip_editor_notes,
    )
    non_ebert = load_reviews(
        args.non_ebert_csv,
        label=0,
        source="non_ebert",
        min_chars=args.min_chars,
        strip_editor_notes=strip_editor_notes,
    )

    ebert_splits = split_by_fractions(
        ebert,
        fractions={
            "generator_train": args.generator_size,
            "classifier_train": args.classifier_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        random_state=args.random_state,
    )
    non_ebert_splits = take_non_ebert_splits(
        non_ebert,
        target_counts={
            "classifier_train": len(ebert_splits["classifier_train"]),
            "val": len(ebert_splits["val"]),
            "test": len(ebert_splits["test"]),
        },
        negative_ratio=args.negative_ratio,
        random_state=args.random_state,
    )

    write_split(ebert_splits["generator_train"], args.output_dir / "ebert_generator_train.csv")
    write_split(ebert_splits["classifier_train"], args.output_dir / "ebert_classifier_train.csv")
    write_split(ebert_splits["val"], args.output_dir / "ebert_val.csv")
    write_split(ebert_splits["test"], args.output_dir / "ebert_test.csv")
    write_split(non_ebert_splits["classifier_train"], args.output_dir / "non_ebert_classifier_train.csv")
    write_split(non_ebert_splits["val"], args.output_dir / "non_ebert_val.csv")
    write_split(non_ebert_splits["test"], args.output_dir / "non_ebert_test.csv")
    write_split(non_ebert_splits["unused"], args.output_dir / "non_ebert_unused.csv")

    classifier_train = mix_classifier_split(
        ebert_splits["classifier_train"],
        non_ebert_splits["classifier_train"],
        random_state=args.random_state,
    )
    classifier_val = mix_classifier_split(
        ebert_splits["val"],
        non_ebert_splits["val"],
        random_state=args.random_state,
    )
    classifier_test = mix_classifier_split(
        ebert_splits["test"],
        non_ebert_splits["test"],
        random_state=args.random_state,
    )

    write_split(classifier_train, args.output_dir / "classifier_train.csv")
    write_split(classifier_val, args.output_dir / "classifier_val.csv")
    write_split(classifier_test, args.output_dir / "classifier_test.csv")

    split_config = {
        "random_state": args.random_state,
        "min_chars": args.min_chars,
        "strip_editor_notes": strip_editor_notes,
        "ebert_fractions": {
            "generator_train": args.generator_size,
            "classifier_train": args.classifier_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        "negative_ratio": args.negative_ratio,
        "notes": [
            "Classifier CSVs are randomly shuffled mixtures of Ebert positives and non-Ebert negatives.",
            "The generator and classifier train on disjoint Ebert reviews.",
            "The Ebert test split is locked for final evaluation.",
        ],
    }
    (args.output_dir / "split_config.json").write_text(
        json.dumps(split_config, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "split": "ebert_generator_train",
                "rows": len(ebert_splits["generator_train"]),
                "purpose": "generator training only",
            },
            {
                "split": "ebert_classifier_train",
                "rows": len(ebert_splits["classifier_train"]),
                "purpose": "classifier positive training rows only",
            },
            {
                "split": "ebert_val",
                "rows": len(ebert_splits["val"]),
                "purpose": "validation and model selection",
            },
            {
                "split": "ebert_test",
                "rows": len(ebert_splits["test"]),
                "purpose": "locked final evaluation",
            },
            {
                "split": "non_ebert_classifier_train",
                "rows": len(non_ebert_splits["classifier_train"]),
                "purpose": "classifier negative training rows",
            },
            {
                "split": "non_ebert_val",
                "rows": len(non_ebert_splits["val"]),
                "purpose": "classifier validation negatives",
            },
            {
                "split": "non_ebert_test",
                "rows": len(non_ebert_splits["test"]),
                "purpose": "classifier test negatives",
            },
            {
                "split": "non_ebert_unused",
                "rows": len(non_ebert_splits["unused"]),
                "purpose": "held back from the main balanced classifier",
            },
            {
                "split": "classifier_train",
                "rows": len(classifier_train),
                "purpose": "randomly mixed classifier training rows",
            },
            {
                "split": "classifier_val",
                "rows": len(classifier_val),
                "purpose": "randomly mixed classifier validation rows",
            },
            {
                "split": "classifier_test",
                "rows": len(classifier_test),
                "purpose": "randomly mixed locked classifier test rows",
            },
        ]
    )
    write_split(summary, args.output_dir / "split_summary.csv")


if __name__ == "__main__":
    main()
