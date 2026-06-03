#!/usr/bin/env python3
"""Generate zero-shot and few-shot GPT baselines through the OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


PLOT_RE = re.compile(r"Wikipedia plot:\n(?P<plot>.*?)\n\nReview:", re.DOTALL)


SYSTEM_PROMPT = (
    "You are writing film criticism in a style inspired by Roger Ebert: clear, humane, "
    "observant, witty when appropriate, and focused on how the movie works as a movie. "
    "Do not claim to be Roger Ebert. Do not mention that you are imitating a style. "
    "Use the plot as grounding, but write a review with critical judgment rather than a plot summary."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPT zero-shot/few-shot review baselines.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/modeling/generator_train.jsonl"))
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/modeling/generator_test.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/gpt_baseline_generated_reviews.jsonl"))
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument(
        "--shot-strategy",
        choices=["similar", "random"],
        default="similar",
        help="How to choose few-shot examples from the training set.",
    )
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-plot-chars", type=int, default=3500)
    parser.add_argument("--max-shot-plot-chars", type=int, default=1000)
    parser.add_argument("--max-shot-review-chars", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output instead of resuming completed URLs.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def shorten(text: str, max_chars: int) -> str:
    text = clean(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def extract_plot(prompt: str) -> str:
    match = PLOT_RE.search(prompt)
    return clean(match.group("plot")) if match else ""


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def completed_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    urls = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = record.get("metadata", {}).get("url")
            if url:
                urls.add(url)
    return urls


def movie_header(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    lines = [f"Movie: {metadata.get('title', '')}"]
    if metadata.get("movie_year"):
        lines.append(f"Year: {metadata['movie_year']}")
    if metadata.get("rating"):
        lines.append(f"Roger Ebert rating: {metadata['rating']} / 4")
    return "\n".join(lines)


def target_prompt(record: dict[str, Any], max_plot_chars: int) -> str:
    return (
        movie_header(record)
        + "\nWikipedia plot:\n"
        + shorten(extract_plot(record["prompt"]), max_plot_chars)
        + "\n\nWrite the review:"
    )


def format_example(record: dict[str, Any], max_plot_chars: int, max_review_chars: int, index: int) -> str:
    return (
        f"Example {index}\n"
        + movie_header(record)
        + "\nWikipedia plot:\n"
        + shorten(extract_plot(record["prompt"]), max_plot_chars)
        + "\nReview:\n"
        + shorten(record["completion"], max_review_chars)
    )


def choose_random_examples(train_records: list[dict[str, Any]], shots: int, rng: random.Random) -> list[dict[str, Any]]:
    if shots <= 0:
        return []
    return rng.sample(train_records, k=min(shots, len(train_records)))


class SimilarExampleRetriever:
    def __init__(self, train_records: list[dict[str, Any]]) -> None:
        self.train_records = train_records
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            [extract_plot(record["prompt"]) for record in train_records]
        )

    def choose(self, record: dict[str, Any], shots: int) -> list[dict[str, Any]]:
        if shots <= 0:
            return []
        query = self.vectorizer.transform([extract_plot(record["prompt"])])
        scores = cosine_similarity(query, self.matrix).ravel()
        indices = scores.argsort()[-shots:][::-1]
        return [self.train_records[int(index)] for index in indices]


def build_messages(
    record: dict[str, Any],
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    if examples:
        example_text = "\n\n".join(
            format_example(
                example,
                max_plot_chars=args.max_shot_plot_chars,
                max_review_chars=args.max_shot_review_chars,
                index=index,
            )
            for index, example in enumerate(examples, start=1)
        )
        user_content = (
            "Here are examples of the desired review style and structure:\n\n"
            + example_text
            + "\n\nNow write a new review for the target movie. "
            + "Do not copy sentences from the examples. Do not summarize mechanically.\n\n"
            + target_prompt(record, args.max_plot_chars)
        )
    else:
        user_content = (
            "Write a film review in a Roger Ebert-inspired critical style for the target movie. "
            "Ground the review in the plot, but prioritize criticism, interpretation, and judgment.\n\n"
            + target_prompt(record, args.max_plot_chars)
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return clean(text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return clean("\n".join(chunks))


def call_openai(client: OpenAI, messages: list[dict[str, str]], args: argparse.Namespace) -> tuple[str, str]:
    response = client.responses.create(
        model=args.model,
        instructions=messages[0]["content"],
        input=messages[1]["content"],
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        store=False,
    )
    return response_text(response), response.id


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running GPT baselines.")

    rng = random.Random(args.seed)
    train_records = load_jsonl(args.train_jsonl)
    test_records = load_jsonl(args.input_jsonl, limit=args.limit)
    retriever = SimilarExampleRetriever(train_records) if args.shot_strategy == "similar" else None

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    done = completed_urls(args.output_jsonl)
    write_header_mode = "a"
    client = OpenAI()

    with args.output_jsonl.open(write_header_mode, encoding="utf-8") as sink:
        for record in tqdm(test_records, desc=f"GPT {args.mode}"):
            metadata = record.get("metadata", {})
            url = metadata.get("url", "")
            if url in done:
                continue

            if args.mode == "few_shot":
                if retriever:
                    examples = retriever.choose(record, args.shots)
                else:
                    examples = choose_random_examples(train_records, args.shots, rng)
            else:
                examples = []

            messages = build_messages(record, examples, args)
            generated, response_id = call_openai(client, messages, args)
            output_record = {
                "metadata": {
                    **metadata,
                    "baseline_mode": args.mode,
                    "shot_strategy": args.shot_strategy if args.mode == "few_shot" else "",
                    "shots": len(examples),
                    "model_name": args.model,
                    "response_id": response_id,
                },
                "prompt": messages[-1]["content"],
                "system_prompt": messages[0]["content"],
                "reference_review": record["completion"],
                "generated_review": generated,
            }
            sink.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            sink.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote generations: {args.output_jsonl}")


if __name__ == "__main__":
    main()
