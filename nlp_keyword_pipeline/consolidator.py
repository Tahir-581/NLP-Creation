"""Canonicalization, merge near-duplicates (RapidFuzz + lemma fold), fragment demotion."""

from __future__ import annotations

import copy
import logging
import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from nlp_keyword_pipeline.domain_lexicon import BREED_ALIASES, canonical_breed_phrase
from nlp_keyword_pipeline.models import Candidate

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def _clone_candidate(c: Candidate) -> Candidate:
    return Candidate(
        surface_form=c.surface_form,
        normalized_form=c.normalized_form,
        source_methods=set(c.source_methods),
        source_docs=set(c.source_docs),
        docs_in_title=set(c.docs_in_title),
        docs_in_heading=set(c.docs_in_heading),
        docs_in_body=set(c.docs_in_body),
        docs_in_intro=set(c.docs_in_intro),
        title_hits=c.title_hits,
        heading_hits=c.heading_hits,
        body_hits=c.body_hits,
        intro_hits=c.intro_hits,
        is_entity=c.is_entity,
        from_lexicon=c.from_lexicon,
        is_question=c.is_question,
        notes=copy.copy(c.notes),
        audit_trail=[],
        cluster_id=None,
        cluster_representative=None,
        cluster_label=None,
        canonical_form=c.canonical_form,
        display_form=c.display_form,
        hygiene_result=c.hygiene_result,
        hygiene_reasons=copy.copy(c.hygiene_reasons),
        extractor_confidence=dict(c.extractor_confidence),
        penalties=dict(c.penalties),
    )


def _lemma_fold_phrase(nlp: Language, norm: str) -> str:
    if not norm or not norm.strip():
        return norm
    doc = nlp(norm[: min(len(norm), nlp.max_length)])
    return " ".join(
        t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct
    )


def apply_canonical_forms(registry: dict[str, Candidate], nlp: Language | None) -> None:
    """Set canonical_form / display_form; optionally fold lemma + breed aliases."""
    for c in registry.values():
        n = c.normalized_form
        alias = BREED_ALIASES.get(n)
        if alias:
            c.canonical_form = alias
            c.display_form = c.surface_form or alias.title()
            c.notes.append(f"breed_alias:{n}->{alias}")
            continue
        cb = canonical_breed_phrase(n)
        if cb != n:
            c.canonical_form = cb
            c.display_form = c.surface_form or cb
            c.notes.append(f"canonical_breed:{n}->{cb}")
            continue
        if nlp is not None:
            folded = _lemma_fold_phrase(nlp, n)
            if folded and folded != n and len(folded.split()) == len(n.split()):
                c.canonical_form = folded
                if not c.display_form:
                    c.display_form = c.surface_form or n
                c.notes.append(f"lemma_fold:{n}->{folded}")
            else:
                c.canonical_form = c.canonical_form or n
                c.display_form = c.display_form or c.surface_form or n
        else:
            c.canonical_form = c.canonical_form or n
            c.display_form = c.display_form or c.surface_form or n


def _similarity(
    nlp: Language,
    a: str,
    b: str,
    threshold: int,
) -> int:
    r1 = fuzz.token_sort_ratio(a, b)
    if r1 >= threshold:
        return r1
    if nlp is None or r1 < threshold - 12:
        return r1
    try:
        la = _lemma_fold_phrase(nlp, a)
        lb = _lemma_fold_phrase(nlp, b)
        r2 = fuzz.token_sort_ratio(la, lb)
        return max(r1, r2)
    except Exception:
        return r1


def consolidate_registry(
    registry: dict[str, Candidate],
    config: PipelineConfig,
    nlp: Language | None = None,
) -> dict[str, Candidate]:
    """
    Apply canonical hints, then greedy clustering by token_sort_ratio >= threshold.
    """
    if not registry:
        return {}
    if nlp is not None:
        apply_canonical_forms(registry, nlp)

    items = list(registry.values())
    threshold = config.rapidfuzz_merge_threshold

    items.sort(
        key=lambda c: (-len(c.source_docs), -len(c.normalized_form), -len(c.surface_form))
    )

    clusters: list[Candidate] = []
    for c in items:
        merged_into = None
        for rep in clusters:
            if nlp is not None:
                r = _similarity(nlp, c.normalized_form, rep.normalized_form, threshold)
            else:
                r = fuzz.token_sort_ratio(c.normalized_form, rep.normalized_form)
            if r >= threshold:
                merged_into = rep
                break
        if merged_into is not None:
            merged_into.merge_from(c)
        else:
            clusters.append(_clone_candidate(c))

    out = {c.normalized_form: c for c in clusters}
    logger.info("Consolidation: %d -> %d candidates", len(registry), len(out))
    return out


def apply_fragment_demotions(
    registry: dict[str, Candidate],
    config: PipelineConfig,
) -> None:
    """
    After scoring: demote single-token fragments that appear as a token inside a stronger phrase.
    """
    cands = list(registry.values())
    cands.sort(key=lambda x: -x.score)
    strong = [
        c
        for c in cands
        if len(c.normalized_form.split()) >= 2
        and c.score >= config.fragment_demotion_score_threshold
    ]
    if not strong:
        return
    for c in cands:
        if len(c.normalized_form.split()) != 1:
            continue
        if c.from_lexicon or (c.is_entity and len(c.normalized_form) >= 5):
            continue
        tok = re.escape(c.normalized_form)
        pat = re.compile(rf"(?<!\w){tok}(?!\w)")
        for s in strong:
            if pat.search(s.normalized_form) and s.score > c.score:
                c.score = min(c.score, 0.12)
                c.notes.append("fragment_subsumed_by_longer_phrase")
                break
