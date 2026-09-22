"""Command-line interface: summarize a text file with TF-IDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .summarizer import summarize


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tfidf-sum",
        description="Extractive TF-IDF summarizer for Ukrainian and English text.",
    )
    p.add_argument("input", type=Path, help="input .txt file (UTF-8)")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--ratio", type=float, default=0.3, help="fraction of sentences to keep (default: 0.3)"
    )
    group.add_argument(
        "--sentences", type=_positive_int, default=None, help="exact number of sentences to keep"
    )
    p.add_argument(
        "--lang",
        choices=["auto", "uk", "en"],
        default="auto",
        help="text language (default: auto-detect)",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="score by mean token weight (removes long-sentence bias)",
    )
    p.add_argument(
        "--top-terms",
        type=int,
        default=0,
        metavar="N",
        help="also print the N highest-weighted terms",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="write the summary to a file instead of stdout only",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        text = args.input.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 1
    if not text.strip():
        print("error: input file is empty", file=sys.stderr)
        return 1

    try:
        result = summarize(
            text,
            ratio=args.ratio,
            n_sentences=args.sentences,
            lang=args.lang,
            normalize=args.normalize,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.text)
    print(
        f"\n[{len(result.sentences)} sentences, compression: {result.compression:.0%}]",
        file=sys.stderr,
    )

    if args.top_terms > 0:
        print("\ntop terms:", file=sys.stderr)
        for term, weight in result.top_terms[: args.top_terms]:
            print(f"  {term:<20} {weight:.3f}", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.text + "\n", encoding="utf-8")
        print(f"written: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
