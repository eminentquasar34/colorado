from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://www.rogerebert.com"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


class BotProtectionError(RuntimeError):
    pass


@dataclass
class ReviewIndexRow:
    title: str
    url: str
    index_author: str | None = None
    index_year: int | None = None
    index_rating: float | None = None


class PageFetcher:
    def get(self, url: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RequestsFetcher(PageFetcher):
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def get(self, url: str) -> str:
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text


class PlaywrightFetcher(PageFetcher):
    def __init__(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `python3 -m pip install -r requirements.txt` "
                "and then `python3 -m playwright install chromium`."
            ) from exc

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"])
        self.page = self.context.new_page()

    def get(self, url: str) -> str:
        self.page.goto(url, wait_until="load", timeout=45_000)
        return self.page.content()

    def close(self) -> None:
        self.context.close()
        self.browser.close()
        self.playwright.stop()


@dataclass
class ReviewDetailRow:
    title: str | None
    review_headline: str | None
    url: str
    author: str | None
    published_date: str | None
    rating: float | None
    mpaa_rating: str | None
    runtime_minutes: int | None
    movie_year: int | None
    genres: str | None
    director: str | None
    cast: str | None
    image_url: str | None
    short_description: str | None
    review_text: str | None


def get_soup(fetcher: PageFetcher, url: str) -> BeautifulSoup:
    html = fetcher.get(url)
    if "security service to protect against malicious bots" in html:
        raise BotProtectionError(
            "RogerEbert.com returned a bot-protection page. Do not bypass this; "
            "use the site only with permission or switch to a licensed/public dataset."
        )
    return BeautifulSoup(html, "lxml")


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"\s+", " ", unescape(text)).strip()


def parse_rating_from_classes(classes: Iterable[str]) -> float | None:
    for class_name in classes:
        match = re.fullmatch(r"star(\d+)", class_name)
        if match:
            return int(match.group(1)) / 10
    return None


def parse_review_cards(soup: BeautifulSoup) -> list[ReviewIndexRow]:
    cards = soup.select("article.review-small-card")
    rows: list[ReviewIndexRow] = []

    for card in cards:
        link = card.select_one("a[href]")
        if link is None:
            continue

        url = urljoin(BASE_URL, link["href"])
        title_node = card.select_one("h3") or link
        title = clean_text(title_node.get_text(" ", strip=True)) or ""

        author = None
        author_node = card.select_one(
            'a[href*="/contributors/"], [rel="author"], .reviewer, .review-author'
        )
        if author_node is not None:
            author = clean_text(author_node.get_text(" ", strip=True))

        year = None
        year_match = re.search(r"-(\d{4})/?$", url)
        if year_match:
            year = int(year_match.group(1))

        rating = None
        for image in card.select("img[class]"):
            classes = image.get("class", [])
            if "filled" in classes:
                rating = parse_rating_from_classes(classes)
                break

        rows.append(
            ReviewIndexRow(
                title=title,
                url=url,
                index_author=author,
                index_year=year,
                index_rating=rating,
            )
        )

    return rows


def parse_review_links(soup: BeautifulSoup) -> list[ReviewIndexRow]:
    rows: list[ReviewIndexRow] = []
    seen: set[str] = set()

    for link in soup.select('a[href^="/reviews/"], a[href^="https://www.rogerebert.com/reviews/"]'):
        url = urljoin(BASE_URL, link["href"])
        if url.rstrip("/") == f"{BASE_URL}/reviews" or url in seen:
            continue
        seen.add(url)
        title = clean_text(link.get_text(" ", strip=True)) or ""
        rows.append(ReviewIndexRow(title=title, url=url))

    return rows


def discover_review_urls(
    fetcher: PageFetcher,
    max_pages: int,
    delay_seconds: float,
    start_page: int = 1,
    reviewer: str | None = None,
    reviewer_id: str | None = None,
    source: str = "reviews",
) -> pd.DataFrame:
    rows: list[ReviewIndexRow] = []

    end_page = start_page + max_pages
    for page_number in tqdm(range(start_page, end_page), desc="Index pages"):
        if source == "reviews":
            query = {}
            if reviewer_id:
                query["_filter_by_reviewer"] = reviewer_id
            if page_number > 1:
                query["_paged"] = str(page_number)
            path = "/reviews"
            if query:
                path = f"{path}?{urlencode(query)}"
        else:
            path = (
                "/contributors/roger-ebert"
                if page_number == 1
                else f"/contributors/roger-ebert/page/{page_number}"
            )

        soup = get_soup(fetcher, urljoin(BASE_URL, path))
        page_rows = parse_review_cards(soup) or parse_review_links(soup)
        if not page_rows:
            break

        if reviewer and not reviewer_id:
            wanted = reviewer.casefold()
            page_rows = [
                row
                for row in page_rows
                if row.index_author and wanted in row.index_author.casefold()
            ]

        rows.extend(page_rows)
        time.sleep(delay_seconds)

    df = pd.DataFrame(asdict(row) for row in rows).drop_duplicates("url")
    return df.reset_index(drop=True)


def first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                return text
    return None


def parse_json_ld(soup: BeautifulSoup) -> dict:
    wanted_types = {"Review", "Movie"}

    def walk_json_ld(data: object) -> dict | None:
        if isinstance(data, list):
            for item in data:
                found = walk_json_ld(item)
                if found:
                    return found
        if isinstance(data, dict):
            item_type = data.get("@type")
            if item_type in wanted_types:
                return data
            for key in ("@graph", "mainEntity", "itemReviewed"):
                found = walk_json_ld(data.get(key))
                if found:
                    return found
        return None

    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            continue
        found = walk_json_ld(data)
        if found:
            return found
    return {}


def json_ld_name(value: object) -> str | None:
    if isinstance(value, dict):
        return clean_text(str(value.get("name") or ""))
    if isinstance(value, str):
        return clean_text(value)
    return None


def json_ld_names(value: object) -> str | None:
    if isinstance(value, list):
        names = [json_ld_name(item) for item in value]
        names = [name for name in names if name]
        return ", ".join(names) if names else None
    return json_ld_name(value)


def parse_iso_duration_minutes(duration: object) -> int | None:
    if not isinstance(duration, str):
        return None
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return hours * 60 + minutes


def parse_header_facts(soup: BeautifulSoup) -> tuple[int | None, str | None, int | None]:
    for node in soup.find_all(["div", "span"]):
        text = clean_text(node.get_text(" ", strip=True))
        if not text or "minutes" not in text:
            continue
        if len(text) > 120:
            continue
        runtime_match = re.search(r"(\d+)\s+minutes", text)
        if not runtime_match:
            continue
        runtime = int(runtime_match.group(1))
        parts = [clean_text(part) for part in re.split(r"\s+‧\s+", text)]
        parts = [part for part in parts if part]
        mpaa = None
        year = None
        if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9-]{1,8}", parts[1]):
            mpaa = parts[1]
        if len(parts) >= 3 and re.fullmatch(r"\d{4}", parts[2]):
            year = int(parts[2])
        return runtime, mpaa, year
    return None, None, None


def parse_detail_page(fetcher: PageFetcher, url: str) -> ReviewDetailRow:
    soup = get_soup(fetcher, url)
    json_ld = parse_json_ld(soup)
    movie = json_ld.get("itemReviewed", {}) if isinstance(json_ld.get("itemReviewed"), dict) else {}

    review_headline = clean_text(str(json_ld.get("name") or "")) or first_text(
        soup, ["h1", '[itemprop="name"]']
    )
    title = json_ld_name(movie) or review_headline
    author = json_ld_name(json_ld.get("author")) or first_text(
        soup,
        [
            'a[href*="/contributors/"]',
            '[rel="author"]',
            ".byline a",
            ".byline",
            ".reviewer",
        ],
    )
    published_date = clean_text(str(json_ld.get("datePublished") or "")) or None
    if not published_date:
        date_node = soup.select_one("time[datetime]")
        if date_node is not None:
            published_date = date_node.get("datetime")
    published_date = published_date or first_text(soup, ["time", ".date"])

    rating = None
    rating_value = json_ld.get("reviewRating", {}).get("ratingValue")
    if rating_value is not None:
        try:
            rating = float(rating_value)
        except (TypeError, ValueError):
            rating = None

    header_runtime, header_mpaa, header_year = parse_header_facts(soup)

    mpaa_rating = first_text(soup, [".mpaa-rating strong", "p.mpaa-rating"]) or header_mpaa
    if mpaa_rating:
        mpaa_rating = re.sub(r"^Rated\s+", "", mpaa_rating).strip()
        if len(mpaa_rating) > 20 or "minutes" in mpaa_rating:
            mpaa_rating = header_mpaa

    runtime_minutes = parse_iso_duration_minutes(movie.get("duration")) or header_runtime
    runtime = first_text(soup, [".running-time strong", "p.running-time"])
    if runtime:
        runtime_match = re.search(r"\d+", runtime)
        if runtime_match:
            runtime_minutes = int(runtime_match.group())

    movie_year = header_year
    date_created = clean_text(str(movie.get("dateCreated") or ""))
    if movie_year is None and date_created:
        year_match = re.search(r"\d{4}", date_created)
        if year_match:
            movie_year = int(year_match.group())

    genres = first_text(soup, [".genres strong", "p.genres"])
    director = json_ld_names(movie.get("director"))
    cast = json_ld_names(movie.get("actor"))
    image = movie.get("image", {})
    image_url = image.get("url") if isinstance(image, dict) else None
    short_description = clean_text(BeautifulSoup(unescape(str(json_ld.get("description") or "")), "lxml").get_text(" "))

    body = (
        soup.select_one('[itemprop="reviewBody"]')
        or soup.select_one(".review-body")
        or soup.select_one(".entry-content")
    )
    paragraphs = []
    if body is not None:
        paragraphs = [
            clean_text(paragraph.get_text(" ", strip=True))
            for paragraph in body.select("p")
        ]
    review_text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)

    return ReviewDetailRow(
        title=title,
        review_headline=review_headline,
        url=url,
        author=author,
        published_date=published_date,
        rating=rating,
        mpaa_rating=mpaa_rating,
        runtime_minutes=runtime_minutes,
        movie_year=movie_year,
        genres=genres,
        director=director,
        cast=cast,
        image_url=image_url,
        short_description=short_description,
        review_text=review_text or None,
    )


def scrape_details(
    fetcher: PageFetcher,
    urls: Iterable[str],
    delay_seconds: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for url in tqdm(list(urls), desc="Review pages"):
        try:
            row = parse_detail_page(fetcher, url)
            rows.append(asdict(row))
        except Exception as exc:
            rows.append({"url": url, "error": str(exc)})
        time.sleep(delay_seconds)
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    df.to_json(output_dir / f"{stem}.jsonl", orient="records", lines=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape RogerEbert.com review metadata.")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reviewer", default="Roger Ebert")
    parser.add_argument(
        "--reviewer-id",
        default="222",
        help="FacetWP reviewer id for Roger Ebert on the all-reviews page.",
    )
    parser.add_argument(
        "--permission-confirmed",
        action="store_true",
        help="Confirm you have permission to collect content from RogerEbert.com.",
    )
    parser.add_argument(
        "--source",
        choices=["contributor", "reviews"],
        default="reviews",
        help="Use the all-reviews index with reviewer filter or Roger Ebert's contributor archive.",
    )
    parser.add_argument(
        "--fetcher",
        choices=["playwright", "requests"],
        default="playwright",
        help="Playwright is the default because RogerEbert.com may reject plain requests.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only scrape review index cards, not individual review pages.",
    )
    args = parser.parse_args()

    if not args.permission_confirmed:
        parser.error(
            "Pass --permission-confirmed only after you have permission to collect "
            "content from RogerEbert.com."
        )

    fetcher: PageFetcher
    fetcher = PlaywrightFetcher() if args.fetcher == "playwright" else RequestsFetcher()

    try:
        index_df = discover_review_urls(
            fetcher=fetcher,
            max_pages=args.max_pages,
            delay_seconds=args.delay,
            start_page=args.start_page,
            reviewer=args.reviewer if args.source == "reviews" else None,
            reviewer_id=args.reviewer_id if args.source == "reviews" else None,
            source=args.source,
        )
        write_outputs(index_df, args.output_dir, "ebert_review_index")

        if args.discover_only:
            return

        detail_df = scrape_details(
            fetcher=fetcher,
            urls=index_df["url"].tolist(),
            delay_seconds=args.delay,
        )
        combined_df = index_df.merge(detail_df, on="url", how="left", suffixes=("_index", ""))
        if "title" in combined_df and "title_index" in combined_df:
            combined_df = combined_df.rename(columns={"title": "detail_title"})
            movie_title = combined_df["title_index"].combine_first(combined_df["detail_title"])
            combined_df.insert(0, "title", movie_title)
        if args.reviewer and "author" in combined_df:
            wanted = args.reviewer.casefold()
            combined_df = combined_df[
                combined_df["author"].fillna("").str.casefold().str.contains(wanted)
            ]
        write_outputs(combined_df, args.output_dir, "ebert_reviews")
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
