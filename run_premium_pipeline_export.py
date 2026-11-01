"""Run premium keyword pipeline and export results to JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from nlp_keyword_pipeline.pipeline_premium import run


def flatten_bucket_terms(buckets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten bucketed terms into a single list and preserve bucket name."""
    rows: list[dict[str, Any]] = []
    for bucket_name, terms in (buckets or {}).items():
        for term in terms or []:
            rows.append({"bucket": bucket_name, **term})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dynamic-column CSV from term rows."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run premium pipeline and export premium_result.json, premium_terms.json, premium_terms.csv",
    )
    parser.add_argument(
        "--keyword",
        required=True,
        help="Human-readable keyword/topic phrase, e.g. 'Dog Breeds to Deter Intruders and Keep You Safe'",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source corpus path passed to premium pipeline, e.g. 'examples/synthetic_corpus'",
    )
    parser.add_argument(
        "--output-dir",
        default="output-premium-latest",
        help="Directory for premium output files",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load embedding model only from local cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Delay heavy import so --help is fast and reliable.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, local_files_only=args.local_files_only)
    result = run(args.keyword, model, args.source)

    buckets = result.get("buckets") or {}
    rows = flatten_bucket_terms(buckets)

    (out / "premium_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "premium_terms.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out / "premium_terms.csv", rows)

    print("total_terms", result.get("total_terms"))
    print("output_dir", str(out.resolve()))


if __name__ == "__main__":
    main()
