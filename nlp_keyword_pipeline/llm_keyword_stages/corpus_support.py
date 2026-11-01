"""Heuristic corpus support evaluation with snippets (extends flowchart support_checker ideas)."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from nlp_keyword_pipeline.llm_keyword_stages.models import MergedCandidate, SupportEvaluation

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)

STRONG_DOC_COUNT = 2
FUZZY_WEAK_THRESHOLD = 82
FUZZY_STRONG_THRESHOLD = 95
SNIPPET_RADIUS = 90


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _in_title(term_lower: str, doc: CleanDocument) -> bool:
    return bool(doc.title_text) and term_lower in _normalize(doc.title_text)


def _in_heading(term_lower: str, doc: CleanDocument) -> bool:
    for heading in doc.heading_texts:
        if term_lower in _normalize(heading):
            return True
    for _level, htext in doc.heading_hierarchy:
        if term_lower in _normalize(htext):
            return True
    return False


def _in_heading_or_title(term_lower: str, doc: CleanDocument) -> bool:
    return _in_title(term_lower, doc) or _in_heading(term_lower, doc)


def _exact_match_doc_ids(term_lower: str, docs: list[CleanDocument]) -> list[str]:
    matched: list[str] = []
    for doc in docs:
        if term_lower in _normalize(doc.full_clean_text):
            matched.append(doc.doc_id)
    return matched


def _snippet_around(needle_norm: str, doc: CleanDocument) -> str:
    """First occurrence window in full_clean_text."""
    full = doc.full_clean_text or ""
    low = full.lower()
    idx = low.find(needle_norm)
    if idx == -1:
        return ""
    start = max(0, idx - SNIPPET_RADIUS // 2)
    end = min(len(full), idx + len(needle_norm) + SNIPPET_RADIUS // 2)
    s = full[start:end].strip()
    if start > 0:
        s = "… " + s
    if end < len(full):
        s = s + " …"
    return s


def _best_fuzzy_score(term_lower: str, docs: list[CleanDocument]) -> tuple[int, str]:
    best_score = 0
    best_doc = ""
    for doc in docs:
        candidate_texts = [
            _normalize(doc.title_text),
            _normalize(doc.intro_text),
        ] + [_normalize(h) for _, h in doc.heading_hierarchy]

        for text in candidate_texts:
            if not text:
                continue
            score = fuzz.partial_ratio(term_lower, text)
            if score > best_score:
                best_score = score
                best_doc = doc.doc_id

        if best_score < FUZZY_STRONG_THRESHOLD:
            for sentence in doc.sentences:
                score = fuzz.partial_ratio(term_lower, _normalize(sentence))
                if score > best_score:
                    best_score = score
                    best_doc = doc.doc_id
                if best_score >= FUZZY_STRONG_THRESHOLD:
                    break

    return best_score, best_doc


def _label_and_score(
    exact_count: int,
    single_doc_heading_title: bool,
    fuzzy_score: int,
) -> tuple[float, str, str]:
    """Returns (corpus_support_score, support_label, rationale)."""
    if exact_count >= STRONG_DOC_COUNT:
        return (
            min(1.0, 0.75 + 0.05 * min(exact_count, 5)),
            "strong",
            f"Exact phrase in {exact_count} document(s).",
        )
    if exact_count == 1 and single_doc_heading_title:
        return (0.95, "strong", "Exact phrase in title or heading.")
    if fuzzy_score >= FUZZY_STRONG_THRESHOLD:
        return (0.92, "strong", f"High fuzzy match to corpus (score {fuzzy_score}).")
    if exact_count == 1:
        return (0.55, "weak", "Exact phrase in one document body only.")
    if fuzzy_score >= FUZZY_WEAK_THRESHOLD:
        return (0.45, "weak", f"Partial fuzzy match (score {fuzzy_score}).")
    return (0.0, "none", "No exact or sufficient fuzzy match in corpus.")


def evaluate_support(
    candidate: MergedCandidate,
    docs: list[CleanDocument],
) -> SupportEvaluation:
    term_lower = _normalize(candidate.term)
    if not term_lower or not docs:
        return SupportEvaluation(
            corpus_support_score=0.0,
            support_label="none",
            exact_match_count=0,
            heading_or_title_presence=False,
            supporting_snippets=[],
            rationale="Empty term or empty corpus.",
            in_title=False,
            in_heading=False,
        )

    matched_ids = _exact_match_doc_ids(term_lower, docs)
    exact_count = len(matched_ids)
    doc_map = {d.doc_id: d for d in docs}
    single_in_heading = False
    if exact_count == 1:
        d0 = doc_map.get(matched_ids[0])
        if d0:
            single_in_heading = _in_heading_or_title(term_lower, d0)

    fuzzy_score, fuzzy_doc = _best_fuzzy_score(term_lower, docs)

    score, label, rationale = _label_and_score(
        exact_count,
        single_in_heading,
        fuzzy_score,
    )

    snippets: list[str] = []
    for did in matched_ids[:3]:
        d = doc_map.get(did)
        if d:
            sn = _snippet_around(term_lower, d)
            if sn:
                snippets.append(sn)
    if not snippets and fuzzy_score >= FUZZY_WEAK_THRESHOLD and fuzzy_doc:
        d = doc_map.get(fuzzy_doc)
        if d and d.sentences:
            for sent in d.sentences[:5]:
                if fuzz.partial_ratio(term_lower, _normalize(sent)) >= FUZZY_WEAK_THRESHOLD:
                    snippets.append(sent.strip()[:200])
                    break

    in_title = any(_in_title(term_lower, d) for d in docs)
    in_heading = any(_in_heading(term_lower, d) for d in docs)
    heading_title = in_title or in_heading

    return SupportEvaluation(
        corpus_support_score=score,
        support_label=label,  # type: ignore[arg-type]
        exact_match_count=exact_count,
        heading_or_title_presence=heading_title,
        supporting_snippets=snippets[:5],
        rationale=rationale,
        matched_doc_ids=matched_ids,
        best_fuzzy_score=fuzzy_score,
        in_title=in_title,
        in_heading=in_heading,
    )


def evaluate_support_batch(
    candidates: list[MergedCandidate],
    docs: list[CleanDocument],
) -> list[tuple[MergedCandidate, SupportEvaluation]]:
    return [(c, evaluate_support(c, docs)) for c in candidates]
