#!/usr/bin/env python3
"""Optional transformer cross-encoder for plot/review relevance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: Path, tokenizer: AutoTokenizer, max_length: int, limit: int | None = None) -> None:
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit)
        self.labels = df["label"].astype(int).tolist()
        self.encodings = tokenizer(
            df["text"].astype(str).tolist(),
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optional transformer relevance cross-encoder.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--train-pairs", type=Path, default=Path("models/plot_relevance_classifier_improved/train_pairs.csv"))
    parser.add_argument("--val-pairs", type=Path, default=Path("models/plot_relevance_classifier_improved/validation_pairs.csv"))
    parser.add_argument("--test-pairs", type=Path, default=Path("models/plot_relevance_classifier_improved/test_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/plot_relevance_transformer"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mps", action="store_true")
    return parser.parse_args()


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = PairDataset(args.train_pairs, tokenizer, args.max_length, args.train_limit)
    val_dataset = PairDataset(args.val_pairs, tokenizer, args.max_length, args.val_limit)
    test_dataset = PairDataset(args.test_pairs, tokenizer, args.max_length, args.test_limit)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
