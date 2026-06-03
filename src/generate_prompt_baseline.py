#!/usr/bin/env python3
"""Generate zero-shot and few-shot prompt baselines with a seq2seq model."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PLOT_RE = re.compile(r"Wikipedia plot:\n(?P<plot>.*?)\n\nReview:", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompted baseline reviews.")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/modeling/generator_train.jsonl"))
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/modeling/generator_test.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/prompt_baseline_generated_reviews.jsonl"))
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-mps", action="store_true")
    return parser.parse_args()


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def extract_plot(prompt: str) -> str:
    match = PLOT_RE.search(prompt)
    return clean(match.group("plot")) if match else ""


def shorten(text: str, max_chars: int) -> str:
    text = clean(text)
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def movie_header(record: dict) -> str:
    metadata = record.get("metadata", {})
    lines = [f"Movie: {metadata.get('title', '')}"]
    if metadata.get("movie_year"):
        lines.append(f"Year: {metadata['movie_year']}")
    if metadata.get("rating"):
        lines.append(f"Roger Ebert rating: {metadata['rating']} / 4")
    return "\n".join(lines)


def zero_shot_prompt(record: dict) -> str:
    return record["prompt"]


def few_shot_prompt(record: dict, examples: list[dict]) -> str:
    parts = [
        "Write film reviews in the style of Roger Ebert. Use the movie plot as grounding, "
        "but write critical prose rather than a plot summary."
    ]
    for index, example in enumerate(examples, start=1):
        parts.append(
            "\n\nExample "
            + str(index)
            + "\n"
            + movie_header(example)
            + "\nWikipedia plot:\n"
            + shorten(extract_plot(example["prompt"]), 900)
            + "\nReview:\n"
            + shorten(example["completion"], 900)
        )

    parts.append(
        "\n\nNow write the review for this movie.\n"
        + movie_header(record)
        + "\nWikipedia plot:\n"
        + shorten(extract_plot(record["prompt"]), 1400)
        + "\nReview:"
    )
    return "".join(parts)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() and not args.no_mps else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    train_records = load_jsonl(args.train_jsonl)
    test_records = load_jsonl(args.input_jsonl, limit=args.limit)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as sink:
        for record in tqdm(test_records, desc=f"Generating {args.mode}"):
            if args.mode == "few_shot":
                examples = random.sample(train_records, k=min(args.shots, len(train_records)))
                prompt = few_shot_prompt(record, examples)
            else:
                examples = []
                prompt = zero_shot_prompt(record)

            inputs = tokenizer(
                prompt,
                max_length=args.max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                )
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            sink.write(
                json.dumps(
                    {
                        "metadata": {
                            **record.get("metadata", {}),
                            "baseline_mode": args.mode,
                            "shots": len(examples),
                            "model_name": args.model_name,
                        },
                        "prompt": prompt,
                        "reference_review": record["completion"],
                        "generated_review": generated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote generations: {args.output_jsonl}")


if __name__ == "__main__":
    main()
