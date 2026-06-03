from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlencode, urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from ebert_scraper import (
    BASE_URL,
    PlaywrightFetcher,
    parse_detail_page,
    parse_review_cards,
    parse_review_links,
)


FIELDNAMES = [
    "title",
    "title_index",
    "url",
    "index_author",
    "index_year",
    "index_rating",
    "detail_title",
    "review_headline",
    "author",
    "published_date",
    "rating",
    "mpaa_rating",
    "runtime_minutes",
    "movie_year",
    "genres",
    "director",
    "cast",
    "image_url",
    "short_description",
    "review_text",
    "error",
]


def load_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["url"])["url"].dropna())


def append_row(csv_path: Path, jsonl_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    normalized = {field: row.get(field) for field in FIELDNAMES}
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(normalized)
        csv_file.flush()
    with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        jsonl_file.flush()


def page_path(page_number: int) -> str:
    query = {"_paged": str(page_number)} if page_number > 1 else {}
    return f"/reviews?{urlencode(query)}" if query else "/reviews"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape non-Ebert reviews, appending each row as it is parsed."
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=819)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/non_ebert_reviews"))
    parser.add_argument(
        "--ebert-index",
        type=Path,
        default=Path("data/raw/ebert_full/ebert_review_index.csv"),
        help="Known Ebert URL index to skip before fetching detail pages.",
    )
    parser.add_argument("--permission-confirmed", action="store_true")
    args = parser.parse_args()

    if not args.permission_confirmed:
        parser.error("Pass --permission-confirmed after confirming collection is authorized.")

    csv_path = args.output_dir / "non_ebert_reviews.csv"
    jsonl_path = args.output_dir / "non_ebert_reviews.jsonl"
    completed_urls = load_urls(csv_path)
    ebert_urls = load_urls(args.ebert_index)

    fetcher = PlaywrightFetcher()
    try:
        end_page = args.start_page + args.max_pages
        for page_number in tqdm(range(args.start_page, end_page), desc="Index pages"):
            soup = BeautifulSoup(fetcher.get(urljoin(BASE_URL, page_path(page_number))), "lxml")
            page_rows = parse_review_cards(soup) or parse_review_links(soup)
            if not page_rows:
                break

            for index_row in tqdm(page_rows, desc=f"Page {page_number} reviews", leave=False):
                if index_row.url in ebert_urls or index_row.url in completed_urls:
                    continue

                row = asdict(index_row)
                try:
                    detail = asdict(parse_detail_page(fetcher, index_row.url))
                    if "roger ebert" in str(detail.get("author") or "").casefold():
                        ebert_urls.add(index_row.url)
                        continue
                    row.update(detail)
                    if row.get("title") and row.get("title_index"):
                        row["detail_title"] = row["title"]
                        row["title"] = row["title_index"] or row["detail_title"]
                except Exception as exc:
                    row["error"] = str(exc)

                append_row(csv_path, jsonl_path, row)
                completed_urls.add(index_row.url)
                time.sleep(args.delay)
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
