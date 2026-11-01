"""Heading phrase mining — normalized headings, splits, and short noun-phrase windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

_SPLIT_CHARS = re.compile(r"[\|:,，]+|[–—\-]+")
_LEADING_ENUM = re.compile(
    r"^(\d{1,2}\s*[\.)]\s*|[#*•]+\s*|\(\d+\)\s*)+",
    re.I,
)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']+")
_NAV_MENU = frozenset(
    {
        "home",
        "about",
        "contact",
        "menu",
        "search",
        "share",
        "subscribe",
        "login",
        "sign",
        "more",
        "next",
        "previous",
    }
)


def _menu_like_heading(h: str) -> bool:
    low = h.lower().strip()
    toks = low.split()
    if len(toks) <= 2 and any(t in _NAV_MENU for t in toks):
        return True
    if len(low) < 10 and len(toks) <= 2:
        return True
    return False


def _structural_heading(h: str) -> bool:
    toks = _TOKEN.findall(h)
    return len(toks) >= 4 and not _menu_like_heading(h)


_STOP_HEAD = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "be",
        "your",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
    }
)


def _normalize_heading(raw: str) -> str:
    h = raw.replace("\xa0", " ").strip()
    h = _LEADING_ENUM.sub("", h).strip()
    return " ".join(h.split())


def _maybe_strip_trailing_question(h: str, *, is_full_line: bool) -> str:
    if not h.endswith("?"):
        return h
    low = h.lower()
    q_starts = (
        "what ",
        "which ",
        "who ",
        "whom ",
        "whose ",
        "where ",
        "when ",
        "why ",
        "how ",
        "are ",
        "is ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "will ",
        "have ",
        "has ",
    )
    if any(low.startswith(s) for s in q_starts):
        return h
    if is_full_line:
        return h
    return h[:-1].strip()


def _emit_heading_phrases(
    registry: dict,
    doc_id: str,
    text: str,
    config: PipelineConfig,
    *,
    is_full_line: bool,
) -> None:
    h = _normalize_heading(text)
    if len(h) < 3:
        return
    h = _maybe_strip_trailing_question(h, is_full_line=is_full_line)
    if len(h) < 3:
        return
    method_main = "heading_structural" if _structural_heading(h) else "heading_full"
    register_hit(registry, h, doc_id, method_main, {"heading"}, config=config)

    parts = _SPLIT_CHARS.split(h)
    for part in parts:
        part = part.strip()
        if len(part) >= 8 and part.lower() != h.lower():
            part = _maybe_strip_trailing_question(part, is_full_line=False)
            sm = "heading_structural" if _structural_heading(part) else "heading_split"
            register_hit(registry, part, doc_id, sm, {"heading"}, config=config)

    # Sliding windows of 2–4 content tokens (long headings only; capped per plan)
    toks = _TOKEN.findall(h)
    min_tok = getattr(config, "heading_fragment_min_heading_tokens", 6)
    frag_cap = getattr(config, "heading_fragment_max_per_heading", 14)
    if len(toks) < min_tok:
        return
    lowered = [t.lower() for t in toks]
    emitted = 0
    for win in (2, 3, 4):
        step = 2 if len(toks) > 12 else 1
        for start in range(0, len(toks) - win + 1, step):
            if emitted >= frag_cap:
                return
            slice_toks = toks[start : start + win]
            slice_low = lowered[start : start + win]
            if slice_low[0] in _STOP_HEAD or slice_low[-1] in _STOP_HEAD:
                continue
            frag = " ".join(slice_toks)
            if len(frag) < 10:
                continue
            register_hit(
                registry,
                frag,
                doc_id,
                "heading_fragment",
                {"heading"},
                config=config,
            )
            emitted += 1


def extract(doc: CleanDocument, registry: dict, config: PipelineConfig) -> None:
    for raw in doc.heading_texts:
        if not raw or not raw.strip():
            continue
        _emit_heading_phrases(
            registry,
            doc.doc_id,
            raw,
            config,
            is_full_line=True,
        )
