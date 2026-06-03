#!/usr/bin/env python3
"""Fine-tune a small seq2seq generator on plot-to-Ebert-review examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


class JsonlSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: AutoTokenizer,
        max_source_length: int,
        max_target_length: int,
        limit: int | None = None,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and len(self.examples) >= limit:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                source = record["prompt"]
                target = record["completion"]
                tokenized = tokenizer(
                    source,
                    max_length=max_source_length,
                    truncation=True,
                )
                labels = tokenizer(
                    text_target=target,
                    max_length=max_target_length,
                    truncation=True,
                )["input_ids"]
                tokenized["labels"] = labels
                tokenized["metadata"] = record.get("metadata", {"line_number": line_number})
                self.examples.append(tokenized)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = dict(self.examples[index])
        example.pop("metadata", None)
        return example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Ebert-style generator baseline.")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/modeling/generator_train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("data/modeling/generator_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/ebert_generator_flan_t5_small"))
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-mps", action="store_true", help="Disable Apple Silicon MPS even if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.backends.mps.is_available() and not args.no_mps:
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    train_dataset = JsonlSeq2SeqDataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        limit=args.train_limit,
    )
    val_dataset = JsonlSeq2SeqDataset(
        args.val_jsonl,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        limit=args.val_limit,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        predict_with_generate=False,
        report_to=[],
        fp16=args.fp16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
    }
    try:
        trainer = Seq2SeqTrainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    metrics["train_examples"] = len(train_dataset)
    metrics["val_examples"] = len(val_dataset)
    metrics["model_name"] = args.model_name
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved generator to {args.output_dir}")


if __name__ == "__main__":
    main()
