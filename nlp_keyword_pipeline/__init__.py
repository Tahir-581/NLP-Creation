"""NLP keyword generation from pre-collected competitor JSON pages."""

from __future__ import annotations

from typing import Any

__all__ = ["run_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "run_pipeline":
        from nlp_keyword_pipeline.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
