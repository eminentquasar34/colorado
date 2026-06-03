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

## Create Modeling Splits

Before training the generator or classifier, create fixed splits so the final
evaluation set stays untouched:

```bash
python3 src/create_modeling_splits.py
```

This writes:

- `data/modeling/ebert_generator_train.csv`: Ebert reviews for generator training.
- `data/modeling/ebert_classifier_train.csv`: disjoint Ebert positives for
  classifier training.
- `data/modeling/ebert_val.csv`: validation reviews for model selection and prompt
  tuning.
- `data/modeling/ebert_test.csv`: locked final evaluation reviews. Do not train the
  generator or classifier on this file.
- `data/modeling/classifier_train.csv`, `classifier_val.csv`, and
  `classifier_test.csv`: randomly shuffled mixtures of Ebert positives and
  non-Ebert negatives.

The default Ebert split is 60% generator train, 20% classifier train, 10%
validation, and 10% test. The classifier defaults to a 1:1 class balance by
sampling the same number of non-Ebert negatives as Ebert positives for each
classifier split. Extra non-Ebert reviews are written to
`data/modeling/non_ebert_unused.csv`.

Change the proportions with flags:

```bash
python3 src/create_modeling_splits.py \
  --generator-size 0.70 \
  --classifier-size 0.10 \
  --val-size 0.10 \
  --test-size 0.10
```

## Train The Baseline Style Classifier

Install dependencies, create splits, then train a TF-IDF + logistic regression
baseline:

```bash
python3 -m pip install -r requirements.txt
python3 src/create_modeling_splits.py
python3 src/train_style_classifier.py
```

Artifacts are written to `models/ebert_style_classifier/`:

- `tfidf_logreg.joblib`
- `metrics.json`
- `validation_predictions.csv`
- `test_predictions.csv`

Use this classifier as one style-resemblance signal, not as the whole answer to
whether a generated review is actually faithful to Ebert's likely opinion. The
non-Ebert data comes from RogerEbert.com reviewers after Ebert's death, so a
classifier can learn topic, era, or site-editorial cues unless we control for
them in later experiments.

## Fetch Wikipedia Plot Context

For generator training, first enrich the Ebert splits with movie plot context
from the Kaggle Wikipedia movie plots file:

```bash
python3 src/attach_wikipedia_plots.py \
  --input-csv data/modeling/ebert_generator_train.csv \
  --output-csv data/modeling/ebert_generator_train_with_plots.csv

python3 src/attach_wikipedia_plots.py \
  --input-csv data/modeling/ebert_val.csv \
  --output-csv data/modeling/ebert_val_with_plots.csv

python3 src/attach_wikipedia_plots.py \
  --input-csv data/modeling/ebert_test.csv \
  --output-csv data/modeling/ebert_test_with_plots.csv
```

The join uses normalized movie title plus release year where possible. Any rows
without a confident match keep an empty `wiki_plot` and a `plot_match_status`
explaining why.

You can also fetch live Wikipedia plot sections for missing rows, but this is
slower and subject to rate limits:

```bash
python3 src/fetch_wikipedia_plots.py \
  --input-csv data/modeling/ebert_generator_train.csv \
  --output-csv data/modeling/ebert_generator_train_wikipedia_plots.csv \
  --delay 2 \
  --retries 3
```

The script uses the MediaWiki API, searches by movie title and year, extracts the
`Plot`, `Synopsis`, or `Premise` section when available, and writes each row as it
goes so it can resume after interruption. If a plot section is unavailable, it
falls back to the page intro and marks the row with `status=intro_fallback`.
If Wikipedia returns rate-limit errors, rerun the same command later with a
larger `--delay`; completed `source_url` rows are skipped.

Use the plot text as generator input context. Keep the classifier text-only,
because generated reviews will not contain metadata fields at evaluation time.

Build generator-ready JSONL from the matched plot/review pairs:

```bash
python3 src/build_generator_dataset.py \
  --input-csv data/modeling/ebert_generator_train_with_plots.csv \
  --output-jsonl data/modeling/generator_train.jsonl
```

Train a separate plot-grounding baseline:

```bash
python3 src/train_plot_relevance_classifier.py
```

This relevance classifier sees `wiki_plot + review_text` pairs and learns to
distinguish correct pairs from hard negative pairs. The default setup creates
three wrong-plot negatives per real review:

- a random wrong plot
- a wrong plot from the same decade
- a wrong plot from the same genre

It combines TF-IDF text features with explicit overlap features such as token
Jaccard similarity, title mentions, and plot n-gram copy-rate. It reports both
binary metrics and ranking metrics, where each review is evaluated as one
correct plot against several wrong plots.

Current improved relevance baseline:

```text
Accuracy:       0.939
F1:             0.880
ROC-AUC:        0.978
Top-1 ranking:  0.962
MRR:            0.979
Recall@3:       0.996
```

For a slower optional hard-negative run, add TF-IDF-nearest plots:

```bash
python3 src/train_plot_relevance_classifier.py \
  --negative-types random,same_decade,same_genre,nearest_tfidf
```

An optional transformer cross-encoder is also available:

```bash
python3 src/train_plot_relevance_transformer.py
```

Use the relevance classifier separately from the Ebert style classifier.

## Train A Generator Baseline

Fine-tune a small FLAN-T5 generator:

```bash
python3 src/train_generator.py \
  --output-dir models/ebert_generator_flan_t5_small \
  --epochs 1 \
  --max-source-length 768 \
  --max-target-length 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

On a CPU-only machine, start with a pilot run:

```bash
python3 src/train_generator.py \
  --output-dir models/ebert_generator_pilot \
  --train-limit 200 \
  --val-limit 50 \
  --epochs 1 \
  --max-source-length 512 \
  --max-target-length 256 \
  --batch-size 1 \
  --gradient-accumulation-steps 4
```

Generate and score held-out reviews:

```bash
python3 src/generate_reviews.py \
  --model-dir models/ebert_generator_pilot \
  --input-jsonl data/modeling/generator_test.jsonl \
  --output-jsonl outputs/pilot_generated_reviews.jsonl \
  --limit 25

python3 src/evaluate_generated_reviews.py \
  --generations-jsonl outputs/pilot_generated_reviews.jsonl \
  --output-csv outputs/pilot_generated_review_scores.csv
```

The generated-review scorer reports style probability, plot-match probability,
plot copy-rate, and BERTScore precision/recall/F1 against the held-out real
Ebert review.

Run zero-shot and few-shot prompt baselines:

```bash
python3 src/generate_prompt_baseline.py \
  --mode zero_shot \
  --output-jsonl outputs/zero_shot_flan_t5_small_reviews.jsonl \
  --limit 25

python3 src/generate_prompt_baseline.py \
  --mode few_shot \
  --shots 3 \
  --output-jsonl outputs/few_shot_flan_t5_small_reviews.jsonl \
  --limit 25
```

Run GPT API baselines:

```bash
export OPENAI_API_KEY="..."

python3 src/generate_gpt_baseline.py \
  --mode zero_shot \
  --model gpt-4.1-mini \
  --output-jsonl outputs/gpt_4_1_mini_zero_shot_reviews.jsonl \
  --limit 50

python3 src/generate_gpt_baseline.py \
  --mode few_shot \
  --shots 3 \
  --shot-strategy similar \
  --model gpt-4.1-mini \
  --output-jsonl outputs/gpt_4_1_mini_few_shot_reviews.jsonl \
  --limit 50
```

The GPT script writes one JSONL row at a time and skips completed review URLs
when rerun, so it can resume after interruption.
