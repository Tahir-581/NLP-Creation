"""LLM Flowchart Pipeline package.

Implements the signal-extraction → category-inference → dual-source →
corpus-support-check → ranking flowchart.

Quick start:
    from LLM_FLOWCHART_PIPELINE.pipeline import run
    result = run(keyword="...", embedding_model=model, source="corpus.json")
"""

from __future__ import annotations
