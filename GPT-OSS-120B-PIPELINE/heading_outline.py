"""Per-page heading outline: h1/h2/h3 buckets, boilerplate stripping, LLM level merge."""

from __future__ import annotations

from typing import Any

from nlp_keyword_pipeline.config import PipelineConfig

# Mirror nlp_keyword_pipeline.extractors.headings
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

# Single-token or obvious nav labels (exact line, case-insensitive)
_SINGLE_NAV_LABELS = frozenset(
    {
        "articles",
        "categories",
        "search",
        "notifications",
        "read more",
    }
)

_BOILERPLATE_EXACT = frozenset(
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

_PROMO_SUBSTRINGS = (
    "sign up today",
    "get the latest from",
    "notifications",
)


def _menu_like_heading(h: str) -> bool:
    """Nav-only short lines (do not use the broad 'any short 2-token' rule — that drops real titles like 'Akita')."""
    low = h.lower().strip()
    toks = low.split()
    if len(toks) <= 2 and any(t in _NAV_MENU for t in toks):
        return True
    return False


def is_boilerplate_or_nav_heading(
    text: str,
    config: PipelineConfig | None = None,
) -> bool:
    """True if heading looks like nav, promo, or boilerplate (drop from outline)."""
    raw = (text or "").replace("\xa0", " ").strip()
    if not raw or len(raw) < 2:
        return True

    low = raw.lower()
    if low in _BOILERPLATE_EXACT:
        return True
    if low in _SINGLE_NAV_LABELS:
        return True

    cfg = config or PipelineConfig()
    for pat in cfg.junk_phrases:
        if pat and pat in low:
            return True
    for pat in cfg.blocked_title_patterns:
        if pat and pat in low:
            return True

    for sub in _PROMO_SUBSTRINGS:
        if sub in low:
            return True
    # Newsletter / CTA all-caps blurbs
    if "sign up" in low and "today" in low and len(raw) > 25:
        return True

    if _menu_like_heading(raw):
        return True

    return False


def filter_heading_hierarchy(
    hierarchy: list[tuple[int, str]],
    config: PipelineConfig | None = None,
) -> list[tuple[int, str]]:
    """Drop boilerplate/nav rows; preserve order."""
    out: list[tuple[int, str]] = []
    for level, text in hierarchy:
        t = (text or "").replace("\xa0", " ").strip()
        if not t:
            continue
        if is_boilerplate_or_nav_heading(t, config=config):
            continue
        out.append((level, t))
    return out


def _cap_level(level: int) -> int:
    if level < 1:
        return 2
    if level <= 3:
        return level
    return 3


def hierarchy_to_outline(
    hierarchy: list[tuple[int, str]],
    *,
    config: PipelineConfig | None = None,
) -> dict[str, list[str]]:
    """
    Bucket headings into h1/h2/h3. Levels 4–6 map to h3.
    Case-insensitive dedupe: keep first surface form.
    """
    filtered = filter_heading_hierarchy(hierarchy, config=config)
    seen_lower: set[str] = set()
    h1: list[str] = []
    h2: list[str] = []
    h3: list[str] = []

    for level, text in filtered:
        key = text.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        L = _cap_level(level)
        if L == 1:
            h1.append(text)
        elif L == 2:
            h2.append(text)
        else:
            h3.append(text)

    return {"h1": h1, "h2": h2, "h3": h3}


def ordered_unique_headings(
    hierarchy: list[tuple[int, str]],
    *,
    config: PipelineConfig | None = None,
) -> list[str]:
    """Filtered headings in crawl order, case-insensitive dedupe (first surface form kept)."""
    filtered = filter_heading_hierarchy(hierarchy, config=config)
    seen_lower: set[str] = set()
    out: list[str] = []
    for _, text in filtered:
        k = text.lower()
        if k in seen_lower:
            continue
        seen_lower.add(k)
        out.append(text)
    return out


def needs_outline_level_llm(
    hierarchy: list[tuple[int, str]],
    *,
    config: PipelineConfig | None = None,
) -> bool:
    """True when filtered hierarchy is non-empty and every row is level 2 (crawl default)."""
    filtered = filter_heading_hierarchy(hierarchy, config=config)
    if not filtered:
        return False
    return all(level == 2 for level, _ in filtered)


def outline_from_levels(
    ordered_texts: list[str],
    levels: list[int],
) -> dict[str, list[str]]:
    """Build h1/h2/h3 from parallel lists (same length); levels clipped to 1–3."""
    h1: list[str] = []
    h2: list[str] = []
    h3: list[str] = []
    for text, lv in zip(ordered_texts, levels):
        L = max(1, min(3, int(lv)))
        if L == 1:
            h1.append(text)
        elif L == 2:
            h2.append(text)
        else:
            h3.append(text)
    return {"h1": h1, "h2": h2, "h3": h3}


def apply_llm_assignments(
    ordered_headings: list[str],
    assignments: list[dict[str, Any]],
) -> dict[str, list[str]] | None:
    """
    Map each heading to h1/h2/h3 using model assignments.
    Returns None if validation fails (caller falls back to all-h2 outline).
    Each assignment must be {"text": exact str, "level": 1|2|3}; every ordered
    heading must appear exactly once with matching text.
    """
    if not ordered_headings:
        return {"h1": [], "h2": [], "h3": []}

    allowed = list(ordered_headings)
    set_allowed = set(allowed)
    if len(set_allowed) != len(allowed):
        return None

    by_text: dict[str, int] = {}
    for item in assignments:
        if not isinstance(item, dict):
            continue
        t = item.get("text")
        if not isinstance(t, str):
            continue
        if t not in set_allowed:
            continue
        try:
            lv = int(item.get("level", 2))
        except (TypeError, ValueError):
            continue
        if t in by_text:
            return None
        by_text[t] = max(1, min(3, lv))

    if len(by_text) != len(allowed):
        return None
    for h in allowed:
        if h not in by_text:
            return None

    levels = [by_text[h] for h in allowed]
    return outline_from_levels(allowed, levels)


def fallback_all_h2_outline(ordered_headings: list[str]) -> dict[str, list[str]]:
    """When LLM validation fails: put every heading under h2."""
    return {"h1": [], "h2": list(ordered_headings), "h3": []}
