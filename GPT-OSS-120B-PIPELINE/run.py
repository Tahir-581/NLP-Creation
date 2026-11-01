"""Run GPT-OSS-120B keyword pipeline and export results to JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the workspace root and this directory are on the path so all modules are importable
_here = Path(__file__).resolve().parent
_workspace = _here.parent
for _p in (_workspace, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import run


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
        description=(
            "Run GPT-OSS-120B pipeline and export "
            "result.json, heading_outlines.json, terms.json, terms.csv"
        ),
    )
    parser.add_argument(
        "--keyword",
        required=True,
        help="Human-readable keyword/topic phrase, e.g. 'Best guard dog breeds'",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source corpus path passed to pipeline, e.g. 'examples/synthetic_corpus'",
    )
    parser.add_argument(
        "--output-dir",
        default="output-gpt-oss-120b",
        help="Directory for output files (default: output-gpt-oss-120b)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for intent filtering (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--intent-threshold",
        type=float,
        default=0.20,
        help="Intent coherence cosine threshold (default: 0.20)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load embedding model only from local cache",
    )
    parser.add_argument(
        "--outline-llm-levels",
        action="store_true",
        help=(
            "When all headings are crawl-default level, call GPT-OSS-120B to "
            "assign h1/h2/h3 per verbatim heading (validated; fallback: all h2)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Delay heavy import so --help is fast and reliable.
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        args.model, local_files_only=args.local_files_only
    )

    result = run(
        args.keyword,
        embedding_model,
        args.source,
        intent_threshold=args.intent_threshold,
        outline_llm_levels=args.outline_llm_levels,
    )

    buckets = result.get("buckets") or {}
    rows = flatten_bucket_terms(buckets)

    (out / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    outlines = result.get("page_heading_outlines") or []
    (out / "heading_outlines.json").write_text(
        json.dumps(outlines, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "terms.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out / "terms.csv", rows)

    print(f"model:       GPT-OSS-120B")
    print(f"total_terms: {result.get('total_terms')}")
    print(f"corpus_size: {result.get('corpus_size')}")
    print(f"output_dir:  {out.resolve()}")


if __name__ == "__main__":
    main()
