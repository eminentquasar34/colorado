#!/usr/bin/env python3
"""Attach Kaggle Wikipedia movie plots to review split CSVs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join review CSVs with Kaggle Wikipedia movie plots."
    )
    parser.add_argument(
        "--plots-csv",
        type=Path,
        default=Path("data/wiki_movie_plots_deduped.csv"),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/modeling/ebert_generator_train.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/modeling/ebert_generator_train_with_plots.csv"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for match statistics. Defaults to output CSV stem + _report.json.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def normalize_title(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return clean_text(text)


def normalize_people(value: Any) -> set[str]:
    text = clean_text(value).lower()
    if not text or text == "unknown":
        return set()
    parts = re.split(r",|;| and | & ", text)
    return {clean_text(re.sub(r"[^a-z0-9]+", " ", part)) for part in parts if clean_text(part)}


def extract_year(row: pd.Series) -> str:
    for column in ["movie_year", "index_year", "published_date", "url"]:
        if column not in row:
            continue
        match = YEAR_RE.search(clean_text(row[column]))
        if match:
            return match.group(0)
    return ""


def prepare_plots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Release Year", "Title", "Director", "Genre", "Wiki Page", "Plot"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["plot_title_norm"] = df["Title"].map(normalize_title)
    df["plot_year"] = df["Release Year"].fillna("").astype(str).str.extract(r"((?:19|20)\d{2})")[0].fillna("")
    df["plot_directors_norm"] = df["Director"].map(normalize_people)
    df["plot_text"] = df["Plot"].fillna("").map(clean_text)
    df = df[df["plot_title_norm"].ne("") & df["plot_text"].ne("")]
    return df.reset_index(drop=True)


def choose_candidate(row: pd.Series, candidates: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if candidates.empty:
        return None, "no_match"

    review_year = extract_year(row)
    if review_year:
        same_year = candidates[candidates["plot_year"].eq(review_year)]
        if len(same_year) == 1:
            return same_year.iloc[0], "title_year"
        if len(same_year) > 1:
            candidates = same_year

    if "director" in row:
        review_directors = normalize_people(row["director"])
        if review_directors:
            with_director = candidates[
                candidates["plot_directors_norm"].map(lambda names: bool(names & review_directors))
            ]
            if len(with_director) == 1:
                return with_director.iloc[0], "title_director"
            if len(with_director) > 1:
                candidates = with_director

    if len(candidates) == 1:
        return candidates.iloc[0], "unique_title"
    return None, "ambiguous"


def attach_plots(reviews: pd.DataFrame, plots: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    plot_groups = {
        title: group.reset_index(drop=True)
        for title, group in plots.groupby("plot_title_norm", sort=False)
    }

    enriched_rows = []
    statuses: dict[str, int] = {}
    for _, row in reviews.iterrows():
        title_norm = normalize_title(row["title"])
        candidate, status = choose_candidate(row, plot_groups.get(title_norm, pd.DataFrame()))
        statuses[status] = statuses.get(status, 0) + 1

        output = row.to_dict()
        output["plot_match_status"] = status
        if candidate is not None:
            output.update(
                {
                    "wiki_release_year": candidate["Release Year"],
                    "wiki_title": candidate["Title"],
                    "wiki_origin_ethnicity": candidate.get("Origin/Ethnicity", ""),
                    "wiki_director": candidate["Director"],
                    "wiki_cast": candidate.get("Cast", ""),
                    "wiki_genre": candidate["Genre"],
                    "wiki_page": candidate["Wiki Page"],
                    "wiki_plot": candidate["plot_text"],
                }
            )
        else:
            output.update(
                {
                    "wiki_release_year": "",
                    "wiki_title": "",
                    "wiki_origin_ethnicity": "",
                    "wiki_director": "",
                    "wiki_cast": "",
                    "wiki_genre": "",
                    "wiki_page": "",
                    "wiki_plot": "",
                }
            )
        enriched_rows.append(output)

    enriched = pd.DataFrame(enriched_rows)
    report = {
        "rows": int(len(enriched)),
        "matched_rows": int(enriched["wiki_plot"].fillna("").ne("").sum()),
        "match_rate": float(enriched["wiki_plot"].fillna("").ne("").mean()) if len(enriched) else 0.0,
        "statuses": statuses,
    }
    return enriched, report


def main() -> None:
    args = parse_args()
    report_json = args.report_json
    if report_json is None:
        report_json = args.output_csv.with_name(f"{args.output_csv.stem}_report.json")

    reviews = pd.read_csv(args.input_csv)
    if "title" not in reviews.columns:
        raise ValueError(f"{args.input_csv} must include a title column.")

    plots = prepare_plots(args.plots_csv)
    enriched, report = attach_plots(reviews, plots)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output_csv, index=False)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(enriched)} rows: {args.output_csv}")
    print(f"Wrote match report: {report_json}")
    print(f"Matched {report['matched_rows']} / {report['rows']} rows ({report['match_rate']:.1%})")


if __name__ == "__main__":
    main()
