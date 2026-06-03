# Roger Ebert Review Generation Project

This project explores whether language models can generate plot-grounded film
reviews in a Roger Ebert-like critical style for films he did not review.

## Start With The Ebert Scrape

RogerEbert.com's Terms of Use say that site content may not be scraped or copied
without prior written permission. Only run collection after you have that
permission, and pass `--permission-confirmed` to document the authorized run.

Install the scraping dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
```

Run a small smoke test first:

```bash
python3 src/ebert_scraper.py --max-pages 2 --delay 1 --permission-confirmed
```

Scrape in batches for longer runs:

```bash
python3 src/ebert_scraper.py --start-page 1 --max-pages 50 --delay 1 --output-dir data/raw/ebert_pages_001_050 --permission-confirmed
python3 src/ebert_scraper.py --start-page 51 --max-pages 50 --delay 1 --output-dir data/raw/ebert_pages_051_100 --permission-confirmed
```

Combine completed batches:

```bash
python3 src/combine_ebert_batches.py data/raw/ebert_pages_001_050 data/raw/ebert_pages_051_100 --output-dir data/processed
```

Scrape non-Ebert reviews for classifier negatives. This appends rows as it goes
and can resume if `non_ebert_reviews.csv` already exists:

```bash
python3 src/scrape_non_ebert_reviews.py --start-page 1 --max-pages 819 --delay 0.5 --permission-confirmed
```

Outputs are written to:

- `data/raw/ebert_review_index.csv`
- `data/raw/ebert_review_index.jsonl`
- `data/raw/ebert_reviews.csv`
- `data/raw/ebert_reviews.jsonl`

For the full corpus, increase `--max-pages` gradually and keep a polite delay:

```bash
python3 src/ebert_scraper.py --max-pages 500 --delay 1.5 --permission-confirmed
```

The scraper discovers review links from the all-reviews page using the Roger
Ebert reviewer filter, fetches each review page, validates the author, and stores
movie title, review headline, URL, author, date, rating, MPAA rating, runtime,
movie year, director, cast, image URL, short description, and review text.
