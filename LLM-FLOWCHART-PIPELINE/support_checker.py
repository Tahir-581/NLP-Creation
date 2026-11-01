"""Corpus support checker for LLM-generated terms.

Implements the diamond decision node in the flowchart:

    LLM Generated → Corpus support check
        ├── Strong support  → Keep   (support_score = 1.0)
        ├── Weak support    → Demote (support_score = 0.5)
        └── No support      → Reject (support_score = 0.0)

Support logic (all checks are case-insensitive):

  STRONG:
    - Exact substring in 2 or more documents, OR
    - Exact substring in any document's title or heading zone.

  WEAK:
    - Exact substring in exactly 1 document body (not title/heading), OR
    - RapidFuzz partial_ratio >= FUZZY_WEAK_THRESHOLD in any document.

  REJECT:
    - No exact or fuzzy match in any document.

All thresholds are exposed as module-level constants so they can be
overridden without touching the function signatures.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument
    from llm_generator import LLMTerm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

STRONG_DOC_COUNT: int = 2
"""Exact match in this many documents or more → Strong support."""

FUZZY_WEAK_THRESHOLD: int = 82
"""RapidFuzz partial_ratio score at or above this → at least Weak support."""

FUZZY_STRONG_THRESHOLD: int = 95
"""RapidFuzz partial_ratio at or above this → Strong support (even in 1 doc)."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _in_heading_or_title(term_lower: str, doc: CleanDocument) -> bool:
    """True if the term appears in the document's title or any heading."""
    if term_lower in _normalize(doc.title_text):
        return True
    for heading in doc.heading_texts:
        if term_lower in _normalize(heading):
            return True
    # Also check structured heading_hierarchy
    for _level, htext in doc.heading_hierarchy:
        if term_lower in _normalize(htext):
            return True
    return False


def _exact_match_docs(term_lower: str, docs: list[CleanDocument]) -> list[str]:
    """Return doc_ids where term appears as an exact substring in full_clean_text."""
    matched: list[str] = []
    for doc in docs:
        if term_lower in _normalize(doc.full_clean_text):
            matched.append(doc.doc_id)
    return matched


def _best_fuzzy_score(term_lower: str, docs: list[CleanDocument]) -> tuple[int, str]:
    """Return (best_partial_ratio, doc_id) across all docs."""
    best_score = 0
    best_doc = ""
    for doc in docs:
        # Check against title + headings + intro (shorter, faster)
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

        # If still no strong signal, scan sentences in body
        if best_score < FUZZY_STRONG_THRESHOLD:
            for sentence in doc.sentences:
                score = fuzz.partial_ratio(term_lower, _normalize(sentence))
                if score > best_score:
                    best_score = score
                    best_doc = doc.doc_id
                if best_score >= FUZZY_STRONG_THRESHOLD:
                    break

    return best_score, best_doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_support(
    llm_terms: list[LLMTerm],
    docs: list[CleanDocument],
) -> list[LLMTerm]:
    """Classify each LLM-generated term as keep / demote / reject.

    Mutates each LLMTerm in-place (sets support_status, support_score,
    matched_docs) and returns the same list with those fields populated.

    Parameters
    ----------
    llm_terms:
        Output of llm_generator.generate_llm_terms().
    docs:
        Full CleanDocument list used as the support corpus.

    Returns
    -------
    list[LLMTerm]
        Same list with support_status / support_score / matched_docs filled in.
        Terms with support_status = "reject" remain in the list but will be
        excluded from the final ranking by ranker.py.
    """
    if not docs:
        logger.warning(
            "[support_checker] No documents to check against; all terms rejected."
        )
        for term in llm_terms:
            term.support_status = "reject"
            term.support_score = 0.0
        return llm_terms

    keep_count = demote_count = reject_count = 0

    for lt in llm_terms:
        term_lower = _normalize(lt.term)
        if not term_lower:
            lt.support_status = "reject"
            lt.support_score = 0.0
            reject_count += 1
            continue

        # ------------------------------------------------------------------
        # 1. Exact match check
        # ------------------------------------------------------------------
        matched_doc_ids = _exact_match_docs(term_lower, docs)
        exact_count = len(matched_doc_ids)
        lt.matched_docs = matched_doc_ids

        if exact_count >= STRONG_DOC_COUNT:
            lt.support_status = "keep"
            lt.support_score = 1.0
            keep_count += 1
            logger.debug(
                "[support_checker] STRONG (exact %d docs): %s", exact_count, lt.term
            )
            continue

        # Check if the single exact match is in a high-value zone
        if exact_count == 1:
            doc_map = {d.doc_id: d for d in docs}
            matched_doc = doc_map.get(matched_doc_ids[0])
            if matched_doc and _in_heading_or_title(term_lower, matched_doc):
                lt.support_status = "keep"
                lt.support_score = 1.0
                keep_count += 1
                logger.debug(
                    "[support_checker] STRONG (heading/title zone): %s", lt.term
                )
                continue

        # ------------------------------------------------------------------
        # 2. Fuzzy match check
        # ------------------------------------------------------------------
        fuzzy_score, fuzzy_doc_id = _best_fuzzy_score(term_lower, docs)

        if fuzzy_score >= FUZZY_STRONG_THRESHOLD:
            lt.support_status = "keep"
            lt.support_score = 1.0
            if fuzzy_doc_id and fuzzy_doc_id not in lt.matched_docs:
                lt.matched_docs.append(fuzzy_doc_id)
            keep_count += 1
            logger.debug(
                "[support_checker] STRONG (fuzzy %d): %s", fuzzy_score, lt.term
            )
            continue

        if exact_count == 1 or fuzzy_score >= FUZZY_WEAK_THRESHOLD:
            lt.support_status = "demote"
            lt.support_score = 0.5
            if fuzzy_doc_id and fuzzy_doc_id not in lt.matched_docs:
                lt.matched_docs.append(fuzzy_doc_id)
            demote_count += 1
            logger.debug(
                "[support_checker] WEAK (exact=%d, fuzzy=%d): %s",
                exact_count,
                fuzzy_score,
                lt.term,
            )
            continue

        # ------------------------------------------------------------------
        # 3. No support
        # ------------------------------------------------------------------
        lt.support_status = "reject"
        lt.support_score = 0.0
        reject_count += 1
        logger.debug("[support_checker] REJECT: %s", lt.term)

    logger.info(
        "[support_checker] %d terms → keep=%d, demote=%d, reject=%d",
        len(llm_terms),
        keep_count,
        demote_count,
        reject_count,
    )
    return llm_terms
