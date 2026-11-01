"""Tests for rich signal extraction, consolidation, and stat typing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve().parent.parent
_workspace = _here.parent
for _p in (_workspace, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import signal_extractor as se
from nlp_keyword_pipeline.models import CleanDocument
from signal_consolidator import consolidate
from signal_models import (
    ExtractedEntity,
    NumericStat,
    RepeatedConcept,
    RichPageSignals,
)


def test_classify_stat_percent_and_duration():
    st, norm = se._classify_stat_value("15%", "the 15% figure")
    assert st == "percent"
    assert "%" in norm

    st2, _ = se._classify_stat_value("12 years", "live 12 years on average")
    assert st2 == "duration"


def test_classify_stat_range():
    st, norm = se._classify_stat_value("10-15 years", "lifespan 10-15 years")
    assert st == "range"


def test_consolidate_dedupes_concepts_and_entities():
    rich = RichPageSignals(
        title="Small dogs: a guide",
        headings=["Lifespan", "Lifespan"],
        repeated_concepts=[
            RepeatedConcept("dogs", "dog", 5, "many dogs"),
            RepeatedConcept("dog", "dog", 3, "one dog"),
        ],
        extracted_entities=[
            ExtractedEntity("Beagle", "product", 0.8, "the Beagle", "PRODUCT"),
            ExtractedEntity("beagle", "product", 0.7, "a beagle", "PRODUCT"),
        ],
        numeric_stats=[
            NumericStat("15%", "15%", "percent", "about 15% of cases"),
            NumericStat("15%", "15%", "percent", "duplicate"),
        ],
    )
    summary = consolidate(rich)
    assert "Small dogs" in " ".join(summary.title_anchors) or summary.title_anchors
    assert summary.heading_anchors == ["Lifespan"]
    assert len(summary.repeated_concept_anchors) == 1
    assert len(summary.entity_anchors) == 1
    assert len(summary.numeric_stat_anchors) == 1


def test_extract_signals_repeated_concept_smoke():
    doc = CleanDocument(
        doc_id="1",
        url="http://x",
        domain="x",
        title_text="Dog breeds guide",
        description_text="",
        heading_texts=["Small dog breeds"],
        paragraph_texts=["Dogs live many years. Dogs need care."],
        intro_text="Dogs are popular.",
        body_text="Dogs live many years. Dogs need care. Chihuahuas are small dogs.",
        full_clean_text=(
            "Dogs are popular. Dogs live many years. Dogs need care. "
            "Chihuahuas are small dogs."
        ),
        heading_hierarchy=[(2, "Small dog breeds")],
    )
    bundle = se.extract_signals("Dog breeds guide", [doc], top_n_nouns=10, min_noun_count=1)
    assert bundle.rich is not None
    assert bundle.rich.repeated_concepts
    assert isinstance(bundle.named_entities, list)
