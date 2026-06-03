from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Ebert scraper batch outputs.")
    parser.add_argument("batch_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--stem", default="ebert_reviews")
    args = parser.parse_args()

    frames = []
    for batch_dir in args.batch_dirs:
        path = batch_dir / "ebert_reviews.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing batch file: {path}")
        frames.append(pd.read_csv(path))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("url").sort_values(
        ["published_date", "title"], ascending=[False, True], na_position="last"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / f"{args.stem}.csv", index=False)
    combined.to_json(args.output_dir / f"{args.stem}.jsonl", orient="records", lines=True)

    print(f"Wrote {len(combined):,} unique reviews to {args.output_dir}")


if __name__ == "__main__":
    main()
