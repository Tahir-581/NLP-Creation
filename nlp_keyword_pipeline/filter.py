"""Document filtering and near-duplicate removal."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.domain_lexicon import DOMAIN_LEXICON
from nlp_keyword_pipeline.models import CleanDocument, PageDocument
from nlp_keyword_pipeline.normalizer import normalize_text_for_dup

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _word_count_text(text: str) -> int:
    return len(_WS.split(text.strip())) if text.strip() else 0


def _meaningful_headings(doc: PageDocument) -> list[str]:
    return [h for h in doc.headings if h and h.strip()]


def exclusion_reason(doc: PageDocument, config: PipelineConfig) -> str | None:
    if doc.domain in config.excluded_domains:
        return "excluded_domain"
    if config.exclude_reddit_by_default and "reddit.com" in doc.domain:
        return "excluded_forum_reddit"
    if getattr(doc, "has_error_signals", False):
        return "blocked_error_signals"
    if not getattr(doc, "usable_text_fields", None):
        return "missing_usable_text_fields"
    title_lower = (doc.title or "").lower()
    if any(pat in title_lower for pat in config.blocked_title_patterns):
        return "blocked_page_title"
    wc = doc.word_count
    if wc is not None and wc < config.min_word_count:
        return "thin_content_word_count"
    elif wc is None:
        combined = _word_count_text(doc.content) or _word_count_text(
            " ".join(doc.paragraphs)
        )
        if combined < config.min_word_count:
            return "thin_content_inferred"
    mh = len(_meaningful_headings(doc))
    content_wc = _word_count_text(doc.content) if doc.content else 0
    if mh < config.min_meaningful_headings and content_wc < config.min_content_words_if_few_headings:
        return "insufficient_structure"
    return None


def compute_source_relevance(
    doc: PageDocument,
    topic_anchor: str,
    config: PipelineConfig,
) -> tuple[float, list[str]]:
    """
    Layer B: continuous relevance score in ~[0, 1] with human-readable reasons.
    Not pass/fail alone; used for doc metadata and weighted prevalence in scoring.
    """
    reasons: list[str] = []
    anchor_norm = normalize_text_for_dup(topic_anchor)
    anchor_tokens = {t for t in anchor_norm.split() if len(t) > 1}
    if not anchor_tokens:
        anchor_tokens = {anchor_norm} if anchor_norm else set()

    title_norm = normalize_text_for_dup(doc.title or "")
    title_tokens = {t for t in title_norm.split() if len(t) > 1}
    overlap = len(anchor_tokens & title_tokens)
    lexical_overlap = min(1.0, overlap / max(len(anchor_tokens), 1) * 1.8)
    if overlap:
        reasons.append(f"title_token_overlap={overlap}")

    heading_blob = normalize_text_for_dup(" ".join(doc.headings))
    heading_tokens = {t for t in heading_blob.split() if len(t) > 1}
    h_align = len(anchor_tokens & heading_tokens)
    heading_alignment = min(1.0, h_align / max(len(anchor_tokens), 1) * 1.4)
    if h_align:
        reasons.append(f"heading_token_overlap={h_align}")

    full = " ".join(
        [doc.title or ""]
        + list(doc.headings)
        + list(doc.paragraphs)
        + [doc.content or ""]
    )
    full_low = normalize_text_for_dup(full)
    lex_hits = 0
    for term in DOMAIN_LEXICON:
        if len(term) < 4:
            continue
        if term in full_low:
            lex_hits += 1
            if lex_hits >= 40:
                break
    topic_lexicon_support = min(1.0, lex_hits / 25.0)
    if lex_hits:
        reasons.append(f"domain_lexicon_hits={lex_hits}")

    mh = len(_meaningful_headings(doc))
    wc = doc.word_count
    if wc is None:
        wc = _word_count_text(doc.content) or _word_count_text(" ".join(doc.paragraphs))
    struct = 0.5 * min(1.0, mh / 8.0) + 0.5 * min(1.0, (doc.paragraph_count or len(doc.paragraphs)) / 25.0)
    structure_quality = min(1.0, struct)
    reasons.append(f"structure_headings={mh}")

    boilerplate_penalty = min(1.0, getattr(doc, "boilerplate_ratio", 0.0) * 1.1)

    raw = (
        0.28 * lexical_overlap
        + 0.24 * heading_alignment
        + 0.22 * topic_lexicon_support
        + 0.26 * structure_quality
        - 0.35 * boilerplate_penalty
    )
    score = max(0.0, min(1.0, raw))
    reasons.append(f"boilerplate_ratio={getattr(doc, 'boilerplate_ratio', 0.0):.3f}")
    return score, reasons


def _dup_corpus_text(doc: PageDocument) -> str:
    from nlp_keyword_pipeline.normalizer import normalize_text_for_dup

    if doc.content and len(doc.content.strip()) > 50:
        return normalize_text_for_dup(doc.content)
    return normalize_text_for_dup(" ".join(doc.paragraphs))


def remove_near_duplicates(
    docs: list[PageDocument],
    excluded: list[dict],
    config: PipelineConfig,
) -> list[PageDocument]:
    if len(docs) < 2:
        return docs
    corpus = [_dup_corpus_text(d) for d in docs]
    if not any(c.strip() for c in corpus):
        return docs
    try:
        vec = TfidfVectorizer(min_df=1, max_features=5000).fit_transform(corpus)
        sims = cosine_similarity(vec)
    except ValueError:
        return docs
    keep = [True] * len(docs)
    thresh = config.near_dup_cosine_threshold
    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if sims[i, j] > thresh:
                keep[j] = False
                excluded.append(
                    {
                        "doc_id": docs[j].doc_id,
                        "url": docs[j].url,
                        "reason": "near_duplicate",
                    }
                )
    return [d for d, k in zip(docs, keep) if k]


def remove_body_prefix_duplicates(
    docs: list[PageDocument],
    excluded: list[dict],
    config: PipelineConfig,
) -> list[PageDocument]:
    """Phase 2: drop pages whose normalized body prefix is almost identical (listicle clones)."""
    if not config.enable_body_prefix_duplicate_filter or len(docs) < 2:
        return docs
    from nlp_keyword_pipeline.normalizer import normalize_text_for_dup

    n = len(docs)
    prefixes: list[str] = []
    lim = config.body_prefix_duplicate_chars
    for d in docs:
        raw = d.content or " ".join(d.paragraphs)
        t = normalize_text_for_dup(raw)[:lim]
        prefixes.append(t)
    keep = [True] * n
    ratio_thresh = config.body_prefix_duplicate_ratio
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if not prefixes[i] or not prefixes[j]:
                continue
            if fuzz.ratio(prefixes[i], prefixes[j]) >= ratio_thresh:
                keep[j] = False
                excluded.append(
                    {
                        "doc_id": docs[j].doc_id,
                        "url": docs[j].url,
                        "reason": "near_duplicate_body_prefix",
                    }
                )
    return [d for d, k in zip(docs, keep) if k]


def filter_documents(
    docs: list[PageDocument],
    config: PipelineConfig,
) -> tuple[list[PageDocument], list[dict]]:
    """Layer A: hard reject. Layer B: score relevance; soft-exclude low scores."""
    kept: list[PageDocument] = []
    excluded: list[dict] = []
    topic = (config.topic_anchor or "").strip()
    for doc in docs:
        reason = exclusion_reason(doc, config)
        if reason:
            excluded.append({"doc_id": doc.doc_id, "url": doc.url, "reason": reason})
            continue
        rel, rel_reasons = compute_source_relevance(doc, topic, config)
        doc.source_relevance_score = rel
        doc.source_relevance_reasons = rel_reasons
        thresh = getattr(config, "source_relevance_min_threshold", 0.15)
        if rel < thresh:
            excluded.append(
                {
                    "doc_id": doc.doc_id,
                    "url": doc.url,
                    "reason": "source_relevance_below_threshold",
                    "source_relevance_score": rel,
                    "source_relevance_reasons": rel_reasons,
                }
            )
            continue
        kept.append(doc)
    kept = remove_near_duplicates(kept, excluded, config)
    kept = remove_body_prefix_duplicates(kept, excluded, config)
    logger.info("After filter: %d kept, %d excluded", len(kept), len(excluded))
    return kept, excluded


def filter_by_intent_coherence(
    documents: list[CleanDocument],
    query: str,
    embedding_model,
    threshold: float = 0.20,
) -> tuple[list[CleanDocument], list[CleanDocument]]:
    """
    Remove documents whose semantic content is misaligned with the query intent.
    Uses cosine similarity between query embedding and a short doc representation.
    """
    from sentence_transformers import util

    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    kept: list[CleanDocument] = []
    rejected: list[CleanDocument] = []

    for doc in documents:
        doc_repr = f"{doc.title_text}. {doc.body_text[:400]}"
        doc_emb = embedding_model.encode(doc_repr, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, doc_emb).item()

        if sim >= threshold:
            kept.append(doc)
        else:
            rejected.append(doc)
            logger.info(
                "[intent_filter] Rejected %s | sim=%.3f",
                doc.url,
                sim,
            )

    return kept, rejected
