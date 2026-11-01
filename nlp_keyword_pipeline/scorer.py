"""Evidence-based scoring: signals, penalties, confidence, final score."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from spacy.lang.en.stop_words import STOP_WORDS

from nlp_keyword_pipeline.models import Candidate
from nlp_keyword_pipeline.semantic_service import (
    fused_semantic_scores,
    lexical_topic_overlap,
)

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument
    from nlp_keyword_pipeline.topic_anchor import TopicAnchor

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _token_count(norm: str) -> int:
    return len(norm.split()) if norm.strip() else 0


def _phrase_quality(norm: str, c: Candidate, config: PipelineConfig) -> float:
    n = _token_count(norm)
    if n <= 0:
        return 0.0
    toks = norm.split()
    if toks[0] in STOP_WORDS or toks[-1] in STOP_WORDS:
        base = 0.55
    else:
        base = 1.0
    if 2 <= n <= 4:
        len_score = 1.0
    elif n == 1:
        len_score = 0.35 if not (c.is_entity or c.from_lexicon) else 0.65
    elif 5 <= n <= 6:
        len_score = 0.72
    else:
        len_score = 0.45
    return max(0.0, min(1.0, base * len_score))


def _method_agreement(c: Candidate) -> float:
    n = len(c.source_methods)
    if n <= 1:
        return 0.0
    return min(1.0, (n - 1) * 0.33)


def _source_consensus(c: Candidate, config: PipelineConfig) -> float:
    return 1.0 if len(c.source_methods) >= config.source_consensus_min_methods else 0.0


def _heading_title_combo(c: Candidate) -> float:
    return 1.0 if len(c.docs_in_heading) > 0 and len(c.docs_in_title) > 0 else 0.0


def _junk_penalty_val(surface: str, norm: str, config: PipelineConfig) -> float:
    low = (surface + " " + norm).lower()
    hits = sum(1 for j in config.junk_phrases if j in low)
    return min(1.0, hits * 0.35)


def _genericity_penalty_val(norm: str, config: PipelineConfig) -> float:
    toks = norm.split()
    if len(toks) == 1 and toks[0] in config.generic_tokens:
        return 1.0
    if len(toks) == 2 and all(t in config.generic_tokens for t in toks):
        return 0.35
    return 0.0


def _doc_authority_map(
    clean_docs: list[CleanDocument],
    config: PipelineConfig,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for d in clean_docs:
        raw = d.metadata.get("authority")
        if raw is None:
            out[d.doc_id] = None
        else:
            try:
                out[d.doc_id] = float(raw)
            except (TypeError, ValueError):
                out[d.doc_id] = None
    return out


def _doc_relevance_map(clean_docs: list[CleanDocument]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in clean_docs:
        out[d.doc_id] = float(d.metadata.get("source_relevance_score", 1.0) or 0.0)
    return out


def _authority_support(
    c: Candidate,
    id_to_auth: dict[str, float | None],
    config: PipelineConfig,
) -> float:
    vals: list[float] = []
    for did in c.source_docs:
        v = id_to_auth.get(did)
        if v is not None:
            vals.append(v)
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    lo = float(config.authority_min_scale)
    hi = float(config.authority_max_scale)
    return max(0.0, min(1.0, (mean - lo) / (hi - lo + 1e-9)))


def _extractor_trust_mix(c: Candidate, config: PipelineConfig) -> float:
    wmap = getattr(config, "extractor_trust_weights", None) or {}
    if not c.source_methods:
        return 0.0
    vals = [float(wmap.get(m, 0.65)) for m in c.source_methods]
    return sum(vals) / len(vals)


def _redundancy_penalty(
    candidates: list[Candidate],
    base_scores: dict[str, float],
) -> dict[str, float]:
    pen: dict[str, float] = {c.normalized_form: 0.0 for c in candidates}
    norms = sorted({c.normalized_form for c in candidates}, key=len, reverse=True)
    for long_p in norms:
        for short_p in norms:
            if long_p == short_p or len(short_p) < 4:
                continue
            if short_p in long_p and len(short_p) + 2 <= len(long_p):
                if base_scores.get(long_p, 0) > base_scores.get(short_p, 0):
                    pen[short_p] = max(pen[short_p], 0.7)
    return pen


def compute_signal_scores(
    c: Candidate,
    n_docs: int,
    sim_anchor: dict[str, float],
    sim_corpus: dict[str, float],
    sim_fused: dict[str, float],
    id_to_auth: dict[str, float | None],
    id_to_rel: dict[str, float],
    config: PipelineConfig,
) -> dict[str, float]:
    norm = c.normalized_form
    doc_prev = len(c.source_docs) / n_docs
    doc_heading = len(c.docs_in_heading) / n_docs
    doc_title = len(c.docs_in_title) / n_docs
    doc_intro = len(c.docs_in_intro) / n_docs
    rel_sum = sum(id_to_rel.get(did, 1.0) for did in c.source_docs)
    src_rel_w_prev = rel_sum / max(n_docs, 1)
    pq = _phrase_quality(norm, c, config)
    ent = 1.0 if (c.is_entity or c.from_lexicon) else 0.0
    sa = float(sim_anchor.get(norm, 0.0))
    sc = float(sim_corpus.get(norm, 0.0))
    sem_v = float(sim_fused.get(norm, 0.0))
    ext_mix = _extractor_trust_mix(c, config)
    return {
        "document_prevalence": doc_prev,
        "heading_presence": doc_heading,
        "title_presence": doc_title,
        "intro_presence": doc_intro,
        "source_relevance_weighted_prevalence": min(1.0, src_rel_w_prev),
        "phrase_quality": pq,
        "entity_bonus": ent,
        "semantic_anchor_similarity": sa,
        "semantic_corpus_similarity": sc,
        "semantic_relevance": sem_v,
        "source_consensus": _source_consensus(c, config),
        "method_agreement": _method_agreement(c),
        "heading_title_combo": _heading_title_combo(c),
        "authority_support": _authority_support(c, id_to_auth, config),
        "extractor_consensus_weighted": ext_mix,
    }


def compute_penalties(
    c: Candidate,
    candidates: list[Candidate],
    base_scores: dict[str, float],
    redundancy: dict[str, float],
    config: PipelineConfig,
) -> dict[str, float]:
    norm = c.normalized_form
    jp = _junk_penalty_val(c.surface_form, norm, config)
    gp = _genericity_penalty_val(norm, config)
    rp = redundancy.get(norm, 0.0)
    sa = c.subscores.get("semantic_anchor_similarity", 0.0) if c.subscores else 0.0
    drift = 0.45 if float(sa) < 0.25 and len(c.source_docs) < 3 else 0.0
    hy = float(c.penalties.get("hygiene_soft", 0.0))
    weak = 0.0
    if len(c.source_docs) == 1 and len(c.source_methods) == 1:
        weak = 0.25
    noise = min(1.0, jp * 0.8)
    return {
        "junk_penalty_raw": jp,
        "genericity_penalty_raw": gp,
        "redundancy_penalty": rp,
        "semantic_drift_penalty": drift,
        "hygiene_soft_penalty": hy,
        "weak_support_penalty": weak,
        "noise_penalty": noise,
        "boilerplate_association_penalty": jp * 0.5,
    }


def compute_confidence(signals: dict[str, float], penalties: dict[str, float]) -> float:
    base = (
        0.28 * signals.get("document_prevalence", 0)
        + 0.22 * signals.get("extractor_consensus_weighted", 0)
        + 0.22 * signals.get("semantic_relevance", 0)
        + 0.18 * signals.get("heading_presence", 0)
        + 0.10 * signals.get("source_relevance_weighted_prevalence", 0)
    )
    pen = (
        0.35 * penalties.get("weak_support_penalty", 0)
        + 0.25 * penalties.get("hygiene_soft_penalty", 0)
        + 0.20 * penalties.get("semantic_drift_penalty", 0)
    )
    return max(0.0, min(1.0, base - pen))


def compute_final_score(
    signals: dict[str, float],
    penalties: dict[str, float],
    config: PipelineConfig,
) -> float:
    w = config.weights
    pos = (
        w.document_prevalence * signals["document_prevalence"]
        + w.heading_presence * signals["heading_presence"]
        + w.title_presence * signals["title_presence"]
        + w.intro_presence * signals["intro_presence"]
        + w.source_relevance_weighted_prevalence
        * signals["source_relevance_weighted_prevalence"]
        + w.phrase_quality * signals["phrase_quality"]
        + w.entity_bonus * signals["entity_bonus"]
        + w.semantic_relevance * signals["semantic_relevance"]
        + w.source_consensus * signals["source_consensus"]
        + w.method_agreement * signals["method_agreement"]
        + w.heading_title_combo * signals["heading_title_combo"]
        + w.authority_support * signals["authority_support"]
        + w.extractor_consensus_weighted * signals["extractor_consensus_weighted"]
    )
    neg = (
        w.redundancy_penalty * penalties["redundancy_penalty"]
        + w.junk_penalty * penalties["junk_penalty_raw"]
        + w.genericity_penalty * penalties["genericity_penalty_raw"]
        + w.semantic_drift_penalty * penalties["semantic_drift_penalty"]
        + w.hygiene_soft_penalty * penalties["hygiene_soft_penalty"]
    )
    return pos - neg


def _build_audit_trail(c: Candidate, w: Any) -> None:
    s = c.subscores
    p = c.penalties
    lines: list[str] = [
        f"final_score={c.score:.4f} confidence={c.confidence:.4f}",
        f"  + document_prevalence: {w.document_prevalence:.2f} × {s.get('document_prevalence', 0):.4f}",
        f"  + source_rel_weighted_prev: {w.source_relevance_weighted_prevalence:.2f} × {s.get('source_relevance_weighted_prevalence', 0):.4f}",
        f"  + intro_presence: {w.intro_presence:.2f} × {s.get('intro_presence', 0):.4f}",
        f"  + extractor_trust_mix: {w.extractor_consensus_weighted:.2f} × {s.get('extractor_consensus_weighted', 0):.4f}",
        f"  + semantic_relevance: {w.semantic_relevance:.2f} × {s.get('semantic_relevance', 0):.4f}",
        f"  − penalties: redundancy={p.get('redundancy_penalty', 0):.2f} drift={p.get('semantic_drift_penalty', 0):.2f} hygiene_soft={p.get('hygiene_soft_penalty', 0):.2f}",
        f"sources: methods={sorted(c.source_methods)} doc_count={len(c.source_docs)}",
    ]
    c.audit_trail = lines


def score_candidates(
    registry: dict[str, Candidate],
    clean_docs: list[CleanDocument],
    config: PipelineConfig,
    topic_anchor: TopicAnchor | None = None,
) -> None:
    _ = topic_anchor  # reserved for future anchor-specific fusion overrides
    n_docs = max(len(clean_docs), 1)
    w = config.weights
    candidates = list(registry.values())
    id_to_auth = _doc_authority_map(clean_docs, config)
    id_to_rel = _doc_relevance_map(clean_docs)

    unique_norms = list({c.normalized_form for c in candidates})
    max_sem = getattr(config, "semantic_max_phrases_to_encode", 0) or len(unique_norms)
    if len(unique_norms) > max_sem:

        def _sem_priority(x: Candidate) -> tuple:
            return (
                -len(x.source_docs),
                -len(x.docs_in_heading),
                -(1 if x.is_entity or x.from_lexicon else 0),
                -len(x.normalized_form),
            )

        sorted_c = sorted(candidates, key=_sem_priority)
        picked: set[str] = set()
        for c in sorted_c:
            picked.add(c.normalized_form)
            if len(picked) >= max_sem:
                break
        encode_list = sorted(picked)
    else:
        encode_list = unique_norms

    sim_anchor, sim_corpus, sim_fused = fused_semantic_scores(
        encode_list, clean_docs, config
    )
    anchor_str = config.topic_anchor
    for n in unique_norms:
        if n not in sim_fused:
            v = lexical_topic_overlap(anchor_str, n)
            sim_anchor[n] = v
            sim_corpus[n] = v
            sim_fused[n] = v

    base_for_redundancy: dict[str, float] = {}
    for c in candidates:
        sig = compute_signal_scores(
            c,
            n_docs,
            sim_anchor,
            sim_corpus,
            sim_fused,
            id_to_auth,
            id_to_rel,
            config,
        )
        c.subscores = sig
        c.subscores["junk_penalty_raw"] = _junk_penalty_val(
            c.surface_form, c.normalized_form, config
        )
        c.subscores["genericity_penalty_raw"] = _genericity_penalty_val(
            c.normalized_form, config
        )
        pos_partial = (
            w.document_prevalence * sig["document_prevalence"]
            + w.heading_presence * sig["heading_presence"]
            + w.semantic_relevance * sig["semantic_relevance"]
        )
        base_for_redundancy[c.normalized_form] = pos_partial
        c.semantic_score = sig["semantic_relevance"] if sig["semantic_relevance"] > 0 else None

    red = _redundancy_penalty(candidates, base_for_redundancy)
    id_to_url = {d.doc_id: d.url for d in clean_docs}

    _sig_keys = (
        "document_prevalence",
        "heading_presence",
        "title_presence",
        "intro_presence",
        "source_relevance_weighted_prevalence",
        "phrase_quality",
        "entity_bonus",
        "semantic_anchor_similarity",
        "semantic_corpus_similarity",
        "semantic_relevance",
        "source_consensus",
        "method_agreement",
        "heading_title_combo",
        "authority_support",
        "extractor_consensus_weighted",
    )

    for c in candidates:
        c.subscores["semantic_anchor_similarity"] = float(
            sim_anchor.get(c.normalized_form, 0.0)
        )
        pens = compute_penalties(c, candidates, base_for_redundancy, red, config)
        for k, v in pens.items():
            c.penalties[k] = v
        c.subscores["redundancy_penalty"] = pens["redundancy_penalty"]
        c.subscores["junk_penalty_raw"] = pens["junk_penalty_raw"]
        c.subscores["genericity_penalty_raw"] = pens["genericity_penalty_raw"]
        c.subscores["semantic_drift_penalty_raw"] = pens["semantic_drift_penalty"]
        sig = {k: float(c.subscores[k]) for k in _sig_keys}
        c.confidence = compute_confidence(sig, pens)
        c.score = compute_final_score(sig, pens, config)
        c.score -= 0.04 * pens.get("weak_support_penalty", 0.0)
        c.source_relevance_weighted_prevalence = sig["source_relevance_weighted_prevalence"]

        if n_docs >= 8:
            c.importance = "high"
        elif n_docs >= 5:
            c.importance = "medium"
        else:
            c.importance = "low"

        c.example_sources = [id_to_url.get(i, i) for i in list(c.source_docs)[:5]]
        _build_audit_trail(c, w)
