"""Text normalization and CleanDocument construction."""

from __future__ import annotations

import re
from typing import Any

from nlp_keyword_pipeline.models import CleanDocument, PageDocument

_FAQ_LINE = re.compile(
    r"(?i)^(how|what|why|when|where|which|who|can|do|does|is|are)\b"
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_DEFAULT_ZONE_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "heading": 2.5,
    "intro": 2.0,
    "faq": 1.8,
    "body": 1.0,
    "meta": 0.5,
}

_BOILERPLATE_LINES = frozenset(
    {
        "advertisement",
        "read more",
        "read more:",
        "subscribe",
        "subscribe now",
        "sign up",
        "sign in",
        "log in",
        "cookie settings",
        "accept cookies",
        "reject all",
        "manage cookies",
        "privacy settings",
        "related posts",
        "you may also like",
        "share this",
        "follow us",
        "leave a comment",
        "comments are closed",
    }
)


def clean_line(text: str) -> str:
    text = (text or "").replace("\xa0", " ").strip()
    low = text.lower()
    if low in _BOILERPLATE_LINES:
        return ""
    return " ".join(text.split())


def strip_leading_determiners(norm: str) -> str:
    """Remove leading a/an/the from normalized phrase for registry keying."""
    s = (norm or "").strip()
    for p in ("the ", "a ", "an "):
        if s.startswith(p):
            s = s[len(p) :].strip()
            break
    return s


def normalize_text_for_dup(text: str) -> str:
    """Lowercase, collapse whitespace for duplicate comparison."""
    text = clean_line(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def _faq_snippets(headings: list[str], paragraphs: list[str]) -> str:
    lines: list[str] = []
    for h in headings:
        if h and _FAQ_LINE.search(h.strip()):
            lines.append(h.strip())
    for p in paragraphs[:12]:
        if p and _FAQ_LINE.search(p.strip()[:80]):
            lines.append(p.strip()[:500])
    return "\n".join(lines)


def _heading_hierarchy(headings: list[str]) -> list[tuple[int, str]]:
    """Use default level 2 unless line looks like 'H1: Title'."""
    out: list[tuple[int, str]] = []
    for h in headings:
        if not h:
            continue
        m = re.match(r"(?i)^h\s*([1-6])\s*[:.\-]\s*(.+)$", h.strip())
        if m:
            out.append((int(m.group(1)), clean_line(m.group(2))))
        else:
            out.append((2, h.strip()))
    return out


def build_clean_documents(docs: list[PageDocument]) -> list[CleanDocument]:
    clean_docs: list[CleanDocument] = []
    for doc in docs:
        headings = [clean_line(h) for h in doc.headings if clean_line(h)]
        paragraphs = [clean_line(p) for p in doc.paragraphs if clean_line(p)]
        intro = " ".join(paragraphs[:3])
        body = " ".join(paragraphs)
        title_t = clean_line(doc.title)
        desc_t = clean_line(doc.description)
        meta_text = desc_t
        faq_text = _faq_snippets(headings, paragraphs)
        parts = [title_t, desc_t, *headings, *paragraphs]
        full_text = "\n".join(p for p in parts if p)
        sentences = _split_sentences(body)
        hierarchy = _heading_hierarchy(headings)
        meta: dict[str, Any] = {
            "word_count": doc.word_count,
            "authority": doc.authority,
            "source_relevance_score": getattr(doc, "source_relevance_score", 0.0),
            "source_relevance_reasons": list(getattr(doc, "source_relevance_reasons", []) or []),
            "boilerplate_ratio": getattr(doc, "boilerplate_ratio", 0.0),
        }
        clean_docs.append(
            CleanDocument(
                doc_id=doc.doc_id,
                url=doc.url,
                domain=doc.domain,
                title_text=title_t,
                description_text=desc_t,
                heading_texts=headings,
                paragraph_texts=paragraphs,
                intro_text=intro,
                body_text=body,
                full_clean_text=full_text,
                metadata=meta,
                zone_weights=dict(_DEFAULT_ZONE_WEIGHTS),
                sentences=sentences,
                heading_hierarchy=hierarchy,
                faq_text=faq_text,
                meta_text=meta_text,
            )
        )
    return clean_docs
