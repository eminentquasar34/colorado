#!/usr/bin/env python3
"""Print a lightweight outline for a LaTeX paper."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^{}]+)\}")
CAPTION_RE = re.compile(r"\\caption\{([^{}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
ENV_RE = re.compile(r"\\(?:begin|end)\{[^{}]+\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a LaTeX document outline.")
    parser.add_argument("tex_file", type=Path)
    return parser.parse_args()


def strip_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = ENV_RE.sub(" ", text)
    text = COMMAND_RE.sub(" ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", strip_latex(text)))


def section_spans(text: str) -> list[tuple[str, str, int, int]]:
    matches = list(HEADING_RE.finditer(text))
    spans = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append((match.group(1), match.group(2), match.end(), end))
    return spans


def main() -> None:
    args = parse_args()
    text = args.tex_file.read_text(encoding="utf-8")
    spans = section_spans(text)
    total_words = word_count(text)

    print(f"{args.tex_file}")
    print(f"Approximate total words: {total_words}")
    print()

    if not spans:
        print("No section headings found.")
        return

    for level, title, start, end in spans:
        body = text[start:end]
        indent = {"section": "", "subsection": "  ", "subsubsection": "    "}[level]
        marker = {"section": "#", "subsection": "##", "subsubsection": "###"}[level]
        print(f"{indent}{marker} {title} ({word_count(body)} words)")

        labels = LABEL_RE.findall(body)
        captions = CAPTION_RE.findall(body)
        for label in labels:
            print(f"{indent}   label: {label}")
        for caption in captions:
            print(f"{indent}   caption: {caption[:120]}")


if __name__ == "__main__":
    main()
