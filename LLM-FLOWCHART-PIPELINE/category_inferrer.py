"""Category inference via LLM: infer topic dimensions from corpus signals.

This module performs a SINGLE async LLM call to GPT-OSS-120B (OpenAI-compatible endpoint, LLM_BASE_URL)
to determine which topic dimensions are relevant for the given title and corpus signals.

Strict rules (from pipeline brief):
  1. Do NOT rely on predefined categories.
  2. Do NOT force the topic into generic buckets.
  3. Infer the topic dimensions from the title and source signals ONLY.
  4. Output includes name, confidence, and rationale per dimension.

The returned categories are later injected into llm_generator.py to anchor
term generation without any predefined vocabulary (using each category's name).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from gpt_oss_client import DEFAULT_TIMEOUT, chat_completion
from signal_extractor import SignalBundle
from signal_models import InferredCategory, TopicSignalSummary

logger = logging.getLogger(__name__)

_DEFAULT_RATIONALE = "Inferred from corpus signals (legacy string format)."


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a topic-analysis assistant.
Your job is to identify the distinct topic DIMENSIONS relevant to an article
given ONLY the signals extracted from the corpus (title, headings, nouns,
entities, stats, and optional consolidated anchors).

Rules you MUST follow:
1. Do NOT invent categories based on generic knowledge.
2. Do NOT use any predefined category list.
3. Derive dimensions ONLY from the provided signals.
4. Each dimension must be a short phrase (1-4 words) that captures a coherent
   aspect of the topic as evidenced in the signals.
5. Return 6-14 dimensions when the evidence supports them.
6. For each dimension, assign confidence (0.0–1.0) and a one-sentence rationale
   citing which signals justify it.
7. Return ONLY valid JSON. No markdown fences, no explanation outside JSON.

Output format:
{"categories": [{"name": "...", "confidence": 0.85, "rationale": "..."}, ...]}
"""


def _build_prompt(signals: SignalBundle, topic_summary: TopicSignalSummary | None) -> str:
    base = (
        "Below are signals extracted from a real corpus for this article title.\n\n"
        + signals.as_compact_text()
    )
    if topic_summary is not None:
        base += "\n\nConsolidated topic anchors (deduplicated):\n"
        base += topic_summary.as_compact_text()
    base += (
        "\n\nBased ONLY on these signals, infer the topic dimensions for this article. "
        'Return JSON: {"categories": [{"name": "...", "confidence": 0.0, '
        '"rationale": "..."}, ...]}'
    )
    return base


# ---------------------------------------------------------------------------
# JSON helpers (mirrors GPT-OSS-120B approach)
# ---------------------------------------------------------------------------


def _strip_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()


def _extract_first_json(raw: str) -> str:
    text = _strip_fences(raw or "")
    if not text:
        return ""
    decoder = json.JSONDecoder()
    for ch in ("{", "["):
        idx = text.find(ch)
        if idx == -1:
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx : idx + end]
        except json.JSONDecodeError:
            continue
    return text


def _parse_inferred_categories(raw: str) -> list[InferredCategory]:
    """Parse LLM response into InferredCategory objects with legacy fallbacks."""
    candidate = _extract_first_json(raw)
    if not candidate:
        logger.warning("[category_inferrer] Empty response from LLM")
        return []
    try:
        data: Any = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("[category_inferrer] Could not parse JSON: %s", candidate[:200])
        return []

    items: list[Any] = []
    if isinstance(data, dict):
        items = (
            data.get("categories")
            or data.get("dimensions")
            or data.get("topics")
            or []
        )
    elif isinstance(data, list):
        items = data
    else:
        items = []

    result: list[InferredCategory] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            result.append(
                InferredCategory(
                    name=item.strip().lower(),
                    confidence=0.55,
                    rationale=_DEFAULT_RATIONALE,
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        name = None
        for key in ("name", "dimension", "category", "topic", "label"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                name = v.strip().lower()
                break
        if not name:
            continue
        try:
            conf = float(item.get("confidence", item.get("score", 0.65)))
        except (TypeError, ValueError):
            conf = 0.65
        conf = max(0.0, min(1.0, conf))
        rationale = item.get("rationale") or item.get("reason") or item.get("evidence")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = _DEFAULT_RATIONALE
        else:
            rationale = rationale.strip()
        result.append(InferredCategory(name=name, confidence=conf, rationale=rationale))

    return result


def category_dimension_names(categories: list[InferredCategory]) -> list[str]:
    """Names only, for llm_generator and backward-compatible JSON."""
    return [c.name for c in categories]


# ---------------------------------------------------------------------------
# Core async function
# ---------------------------------------------------------------------------


async def infer_categories(
    signals: SignalBundle,
    *,
    topic_summary: TopicSignalSummary | None = None,
    client: httpx.AsyncClient | None = None,
    max_retries: int = 2,
) -> list[InferredCategory]:
    """Make a single LLM call to infer topic dimensions from corpus signals."""
    prompt = _build_prompt(signals, topic_summary)

    owns_client = client is None
    _client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    try:
        for attempt in range(max_retries + 1):
            try:
                raw_text = await chat_completion(
                    _client,
                    system=_SYSTEM_PROMPT,
                    user=prompt,
                    max_tokens=1024,
                    temperature=0.2,
                )
                cats = _parse_inferred_categories(raw_text)
                if cats:
                    logger.info(
                        "[category_inferrer] Inferred %d categories: %s",
                        len(cats),
                        [c.name for c in cats],
                    )
                    return cats
                logger.warning(
                    "[category_inferrer] No categories parsed on attempt %d", attempt + 1
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning(
                    "[category_inferrer] LLM request failed (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries:
                    logger.error("[category_inferrer] All retries exhausted; returning []")
                    return []
        return []
    finally:
        if owns_client:
            await _client.aclose()


def infer_categories_sync(
    signals: SignalBundle,
    *,
    topic_summary: TopicSignalSummary | None = None,
) -> list[InferredCategory]:
    """Synchronous wrapper for infer_categories (for testing / one-off use)."""
    import asyncio

    return asyncio.run(infer_categories(signals, topic_summary=topic_summary))
