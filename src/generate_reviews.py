#!/usr/bin/env python3
"""Generate reviews from a fine-tuned seq2seq model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviews for generator JSONL prompts.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/ebert_generator_flan_t5_small"))
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/modeling/generator_test.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/generated_reviews.jsonl"))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-mps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() and not args.no_mps else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open(encoding="utf-8") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as sink:
        for index, line in enumerate(tqdm(source, desc="Generating")):
            if args.limit is not None and index >= args.limit:
                break
            record = json.loads(line)
            inputs = tokenizer(
                record["prompt"],
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
                        "metadata": record.get("metadata", {}),
                        "prompt": record["prompt"],
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
