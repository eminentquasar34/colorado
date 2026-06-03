#!/usr/bin/env python3
"""Fetch Wikipedia plot sections for review datasets using the MediaWiki API."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "colorado-ebert-style-project/0.1 (educational research)"
PLOT_SECTION_RE = re.compile(r"^(plot|synopsis|premise|plot summary)$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


OUTPUT_COLUMNS = [
    "title",
    "movie_year",
    "source_url",
    "wikipedia_title",
    "wikipedia_pageid",
    "wikipedia_url",
    "wikipedia_section",
    "plot_text",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append Wikipedia plot summaries for movies in a review CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/modeling/ebert_generator_train.csv"),
        help="Review CSV with at least title and url columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/modeling/ebert_generator_train_wikipedia_plots.csv"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-retry-sleep", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args()


def request_json(
    session: requests.Session,
    params: dict[str, Any],
    timeout: float,
    retries: int,
    max_retry_sleep: float,
) -> dict[str, Any]:
    for attempt in range(retries + 1):
        response = session.get(API_URL, params=params, timeout=timeout)
        if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else 2.0 * (attempt + 1)
            sleep_for = min(sleep_for, max_retry_sleep)
            time.sleep(sleep_for)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable retry state")


def clean_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_title(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return clean_text(text)


def movie_year(row: pd.Series) -> str:
    for column in ["movie_year", "index_year", "published_date"]:
        value = row.get(column)
        if pd.notna(value):
            match = re.search(r"\b(19|20)\d{2}\b", str(value))
            if match:
                return match.group(0)
    return ""


def search_queries(title: str, year: str) -> list[str]:
    queries = []
    if year:
        queries.extend(
            [
                f'"{title}" {year} film',
                f'{title} {year} film',
                f'{title} film {year}',
            ]
        )
    queries.extend([f'"{title}" film', f"{title} film", title])
    return queries


def score_candidate(target_title: str, year: str, page_title: str) -> int:
    target = normalize_title(target_title)
    candidate = normalize_title(page_title)
    if not target or not candidate:
        return 0

    score = 0
    if candidate == target:
        score += 100
    elif candidate.startswith(f"{target} "):
        score += 85
    elif target in candidate:
        score += 70

    lowered = page_title.lower()
    if "film" in lowered:
        score += 10
    if year and year in lowered:
        score += 5
    return score


def search_page(
    session: requests.Session,
    title: str,
    year: str,
    timeout: float,
    retries: int,
    max_retry_sleep: float,
) -> dict[str, Any] | None:
    seen_pageids = set()
    candidates = []
    for query in search_queries(title, year):
        data = request_json(
            session,
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
            },
            timeout=timeout,
            retries=retries,
            max_retry_sleep=max_retry_sleep,
        )
        for result in data.get("query", {}).get("search", []):
            pageid = result.get("pageid")
            page_title = result.get("title", "")
            if pageid in seen_pageids:
                continue
            seen_pageids.add(pageid)
            score = score_candidate(title, year, page_title)
            if score:
                candidates.append({"pageid": pageid, "title": page_title, "score": score})
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["score"])
    if best["score"] < 70:
        return None
    return {"pageid": best["pageid"], "title": best["title"]}


def find_plot_section(
    session: requests.Session,
    pageid: int,
    timeout: float,
    retries: int,
    max_retry_sleep: float,
) -> dict[str, str] | None:
    data = request_json(
        session,
        {
            "action": "parse",
            "format": "json",
            "pageid": pageid,
            "prop": "sections",
        },
        timeout=timeout,
        retries=retries,
        max_retry_sleep=max_retry_sleep,
    )
    for section in data.get("parse", {}).get("sections", []):
        line = section.get("line", "")
        if PLOT_SECTION_RE.match(line):
            return {"index": section["index"], "line": line}
    return None


def fetch_section_text(
    session: requests.Session,
    pageid: int,
    section_index: str,
    timeout: float,
    retries: int,
    max_retry_sleep: float,
) -> str:
    data = request_json(
        session,
        {
            "action": "parse",
            "format": "json",
            "pageid": pageid,
            "section": section_index,
            "prop": "text",
            "disableeditsection": 1,
        },
        timeout=timeout,
        retries=retries,
        max_retry_sleep=max_retry_sleep,
    )
    html = data.get("parse", {}).get("text", {}).get("*", "")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("sup, table, style, .mw-editsection"):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def fetch_intro_extract(
    session: requests.Session,
    pageid: int,
    timeout: float,
    retries: int,
    max_retry_sleep: float,
) -> str:
    data = request_json(
        session,
        {
            "action": "query",
            "format": "json",
            "pageids": pageid,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
        },
        timeout=timeout,
        retries=retries,
        max_retry_sleep=max_retry_sleep,
    )
    page = data.get("query", {}).get("pages", {}).get(str(pageid), {})
    return clean_text(page.get("extract", ""))


def fetch_plot_for_row(
    session: requests.Session,
    row: pd.Series,
    timeout: float,
    retries: int,
    max_retry_sleep: float,
) -> dict[str, Any]:
    title = str(row["title"]).strip()
    year = movie_year(row)
    page = search_page(
        session,
        title=title,
        year=year,
        timeout=timeout,
        retries=retries,
        max_retry_sleep=max_retry_sleep,
    )
    if not page:
        return {
            "title": title,
            "movie_year": year,
            "source_url": row.get("url", ""),
            "status": "not_found",
            "error": "",
        }

    pageid = int(page["pageid"])
    wikipedia_title = page["title"]
    wikipedia_url = f"https://en.wikipedia.org/wiki/{quote(wikipedia_title.replace(' ', '_'))}"
    section = find_plot_section(
        session,
        pageid=pageid,
        timeout=timeout,
        retries=retries,
        max_retry_sleep=max_retry_sleep,
    )
    if section:
        plot_text = fetch_section_text(
            session,
            pageid=pageid,
            section_index=section["index"],
            timeout=timeout,
            retries=retries,
            max_retry_sleep=max_retry_sleep,
        )
        status = "ok" if plot_text else "empty_plot_section"
        section_name = section["line"]
    else:
        plot_text = fetch_intro_extract(
            session,
            pageid=pageid,
            timeout=timeout,
            retries=retries,
            max_retry_sleep=max_retry_sleep,
        )
        status = "intro_fallback" if plot_text else "no_plot_section"
        section_name = ""

    return {
        "title": title,
        "movie_year": year,
        "source_url": row.get("url", ""),
        "wikipedia_title": wikipedia_title,
        "wikipedia_pageid": pageid,
        "wikipedia_url": wikipedia_url,
        "wikipedia_section": section_name,
        "plot_text": plot_text,
        "status": status,
        "error": "",
    }


def existing_source_urls(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    existing = pd.read_csv(output_csv, usecols=["source_url"])
    return set(existing["source_url"].dropna().astype(str))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if "title" not in df.columns or "url" not in df.columns:
        raise ValueError("--input-csv must include title and url columns.")
    if args.limit is not None:
        df = df.head(args.limit)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_source_urls(args.output_csv)
    write_header = not args.output_csv.exists()

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    with args.output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()

        pending = df[~df["url"].astype(str).isin(completed)]
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Wikipedia plots"):
            try:
                result = fetch_plot_for_row(
                    session,
                    row=row,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_retry_sleep=args.max_retry_sleep,
                )
            except Exception as exc:  # Keep long runs resumable.
                result = {
                    "title": row.get("title", ""),
                    "movie_year": movie_year(row),
                    "source_url": row.get("url", ""),
                    "status": "error",
                    "error": repr(exc),
                }
            writer.writerow({column: result.get(column, "") for column in OUTPUT_COLUMNS})
            handle.flush()
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
