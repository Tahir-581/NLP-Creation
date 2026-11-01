"""LLM-based term generation grounded in inferred topic dimensions.

Makes a single async LLM call to GPT-OSS-120B (OpenAI-compatible endpoint, LLM_BASE_URL) to
generate keyword candidates, using:

  - The page title
  - The inferred topic dimensions (from category_inferrer.py)
  - Snippets from the top corpus documents (grounding context)

All terms produced here carry ``source = "llm_generated"`` and must pass
through support_checker.py (Strong → Keep, Weak → Demote, None → Reject)
before they reach the final ranked list.

Pipeline rules enforced in the prompt:
  1. Terms must be aligned to the inferred dimensions.
  2. Terms must be grounded in the corpus snippets.
  3. Do NOT invent terms that have no evidence in the corpus.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import httpx

from gpt_oss_client import DEFAULT_TIMEOUT, chat_completion

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)

# How many corpus documents to inject as context (keeps prompt size sane)
_MAX_CONTEXT_DOCS = 4
# Max characters per document snippet
_MAX_DOC_CHARS = 2500


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class LLMTerm:
    """A term produced by the LLM generator."""

    term: str
    normalized: str
    dimension: str
    salience: float
    source: str = "llm_generated"

    # Filled in by support_checker.py
    support_status: str = "pending"   # "keep" | "demote" | "reject"
    support_score: float = 0.0

    # For traceability
    matched_docs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "dimension": self.dimension,
            "salience": round(self.salience, 4),
            "source": self.source,
            "support_status": self.support_status,
            "support_score": round(self.support_score, 4),
            "matched_docs": self.matched_docs,
        }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an SEO keyword-generation assistant.
Your task is to generate keyword terms for an article, strictly grounded in:
  (a) the inferred topic dimensions provided, and
  (b) the corpus text snippets provided.

Rules you MUST follow:
1. Every term must map to one of the provided topic dimensions.
2. Every term must be evidenced or strongly implied by the corpus snippets.
3. Do NOT invent terms that have no grounding in the provided text.
4. Prefer exact or near-exact phrases from the corpus over free inventions.
5. Include both short (1-2 word) and longer (3-5 word) phrase variants.
6. salience is a float 0.0–1.0 indicating how central this term is to the topic.
7. Return ONLY valid JSON. No markdown fences, no explanation.
8. The response MUST be a single JSON object (not a bare array) with a "terms" key.

Output format:
{
  "terms": [
    {"term": "...", "dimension": "...", "salience": 0.0}
  ]
}
"""


def _build_corpus_context(docs: list[CleanDocument]) -> str:
    """Concatenate truncated body snippets from top docs for LLM context."""
    parts: list[str] = []
    for i, doc in enumerate(docs[:_MAX_CONTEXT_DOCS]):
        snippet = (doc.intro_text + " " + doc.body_text).strip()[:_MAX_DOC_CHARS]
        parts.append(f"[Source {i + 1}: {doc.title_text}]\n{snippet}")
    return "\n\n".join(parts)


def _build_prompt(
    title: str,
    dimensions: list[str],
    docs: list[CleanDocument],
) -> str:
    dim_list = "\n".join(f"  - {d}" for d in dimensions)
    corpus_context = _build_corpus_context(docs)
    return (
        f"Article title: {title}\n\n"
        f"Topic dimensions to cover:\n{dim_list}\n\n"
        f"Corpus snippets (ground your terms in this text):\n{corpus_context}\n\n"
        "Generate keyword terms aligned to the dimensions and grounded in the corpus. "
        'Return JSON object: {"terms": [{"term": "...", "dimension": "...", "salience": 0.0}, ...]}'
    )


# ---------------------------------------------------------------------------
# JSON helpers
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


def _parse_terms(raw: str) -> list[dict[str, Any]]:
    candidate = _extract_first_json(raw)
    if not candidate:
        return []
    try:
        data: Any = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("[llm_generator] JSON parse failed: %s", candidate[:200])
        return []

    if isinstance(data, dict):
        # Unwrap common wrappers
        for key in ("terms", "keywords", "results", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            # Try to use any list value
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                return []

    if not isinstance(data, list):
        return []

    result: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        term = (item.get("term") or item.get("keyword") or item.get("phrase") or "").strip()
        if not term:
            continue
        dim = str(item.get("dimension") or item.get("category") or item.get("type") or "general").strip().lower()
        try:
            salience = float(item.get("salience") or item.get("score") or item.get("relevance") or 0.5)
        except (TypeError, ValueError):
            salience = 0.5
        salience = max(0.0, min(1.0, salience))
        result.append({"term": term, "dimension": dim, "salience": salience})

    return result


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Core async function
# ---------------------------------------------------------------------------


async def generate_llm_terms(
    title: str,
    dimensions: list[str],
    docs: list[CleanDocument],
    *,
    client: httpx.AsyncClient | None = None,
    max_retries: int = 2,
) -> list[LLMTerm]:
    """Call the LLM to generate keyword candidates grounded in corpus context.

    Parameters
    ----------
    title:
        The article/page title.
    dimensions:
        Topic dimensions inferred by category_inferrer.
    docs:
        CleanDocument list (top docs used as context snippets).
    client:
        Optional httpx.AsyncClient to reuse.
    max_retries:
        Number of retry attempts on transient HTTP errors.

    Returns
    -------
    list[LLMTerm]
        Raw LLM-generated terms (source = "llm_generated").
        All have support_status = "pending" until support_checker runs.
    """
    if not dimensions:
        logger.warning("[llm_generator] No dimensions provided; skipping generation.")
        return []

    prompt = _build_prompt(title, dimensions, docs)

    owns_client = client is None
    _client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    try:
        for attempt in range(max_retries + 1):
            try:
                raw_text = await chat_completion(
                    _client,
                    system=_SYSTEM_PROMPT,
                    user=prompt,
                    max_tokens=2048,
                    temperature=0.3,
                )
                raw_items = _parse_terms(raw_text)
                if raw_items:
                    seen: set[str] = set()
                    result: list[LLMTerm] = []
                    for item in raw_items:
                        norm = _normalize(item["term"])
                        if not norm or norm in seen:
                            continue
                        seen.add(norm)
                        result.append(
                            LLMTerm(
                                term=item["term"],
                                normalized=norm,
                                dimension=item["dimension"],
                                salience=item["salience"],
                            )
                        )
                    logger.info(
                        "[llm_generator] Generated %d unique terms across %d dimensions",
                        len(result),
                        len({t.dimension for t in result}),
                    )
                    return result
                logger.warning(
                    "[llm_generator] No terms parsed on attempt %d", attempt + 1
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning(
                    "[llm_generator] Request failed (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries:
                    logger.error("[llm_generator] All retries exhausted; returning []")
                    return []
        return []
    finally:
        if owns_client:
            await _client.aclose()
