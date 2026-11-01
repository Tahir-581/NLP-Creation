"""Map character offsets in full_clean_text to structural zones."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument


def char_zone(clean_doc: CleanDocument, idx: int) -> str:
    """
    Zones match build order in normalizer: title, description, headings..., paragraphs...
    Joined with newlines between non-empty parts.
    """
    parts: list[tuple[str, str]] = []
    if clean_doc.title_text:
        parts.append(("title", clean_doc.title_text))
    if clean_doc.description_text:
        parts.append(("title", clean_doc.description_text))
    for h in clean_doc.heading_texts:
        parts.append(("heading", h))
    for p in clean_doc.paragraph_texts:
        parts.append(("body", p))

    pos = 0
    for zone, text in parts:
        if not text:
            continue
        if pos > 0:
            pos += 1
        end = pos + len(text)
        if pos <= idx < end:
            return zone
        pos = end
    return "body"
