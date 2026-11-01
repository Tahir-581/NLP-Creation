"""Async LLM keyword extraction via an Ollama-style chat endpoint (LLM_CHAT_URL) (qwen3-coder-next)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)

GAMMA_CHAT_URL = os.environ.get("LLM_CHAT_URL", "http://localhost:11434/api/chat")
GAMMA_MODEL = os.environ.get("LLM_MODEL", "qwen3-coder-next")

# Conservative timeout: fail fast so the pipeline isn't blocked for minutes
_LLM_TIMEOUT = httpx.Timeout(25.0, connect=8.0)
# Max concurrent requests to avoid overwhelming the API
_CONCURRENCY = 4

SYSTEM_PROMPT = """You are an SEO content analyst replicating Google Natural Language API behavior.
Given a competitor page and a target keyword, extract all terms a content writer would need
to achieve topical coverage. Return ONLY valid JSON. No markdown, no explanation."""

CATEGORIES = [
    "anchor_term",
    "health_condition",
    "care_routine",
    "breed_trait",
    "breed_name",
    "temporal",
    "emotional",
    "lifestyle",
    "longevity_factor",
    "quality_of_life",
    "question_phrase",
    "other",
]


def _gamma_api_key() -> str:
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()


def _strip_json_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()


def _assistant_text_from_response(data: Any) -> str:
    """Parse assistant text from gamma / OpenAI-style JSON bodies."""
    if not isinstance(data, dict):
        return ""

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or first.get("delta") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(str(block.get("text", "")))
                            elif "text" in block:
                                parts.append(str(block.get("text", "")))
                    if parts:
                        return "".join(parts)

    msg = data.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, str):
            return c

    for key in ("content", "response", "text", "output", "result"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


async def _chat_completion(
    client: httpx.AsyncClient,
    *,
    system: str,
    user: str,
    max_tokens: int = 1000,
    temperature: float = 0.3,
) -> str:
    payload: dict[str, Any] = {
        "model": GAMMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    key = _gamma_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = await client.post(GAMMA_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = ""
        if exc.response is not None and isinstance(exc.response.text, str):
            body = _truncate(exc.response.text.strip())
        logger.warning(
            "[llm_extractor] HTTP error calling gamma url=%s status=%s error_type=%s body=%r",
            GAMMA_CHAT_URL,
            status,
            type(exc).__name__,
            body,
        )
        raise
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        # Transient network issues — log concisely without full traceback
        logger.debug(
            "[llm_extractor] Request error calling gamma url=%s error_type=%s error=%r",
            GAMMA_CHAT_URL,
            type(exc).__name__,
            exc,
        )
        raise
    except httpx.RequestError as exc:
        logger.warning(
            "[llm_extractor] Request error calling gamma url=%s error_type=%s error=%r",
            GAMMA_CHAT_URL,
            type(exc).__name__,
            exc,
        )
        raise

    try:
        data = resp.json()
    except ValueError as exc:
        logger.warning(
            "[llm_extractor] Invalid JSON from gamma status=%s body=%r error_type=%s error=%r",
            resp.status_code,
            _truncate(resp.text.strip() if isinstance(resp.text, str) else ""),
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise

    text = _assistant_text_from_response(data)
    return text if isinstance(text, str) else str(text or "")


def build_extraction_prompt(keyword: str, page_title: str, body_text: str) -> str:
    cats_str = ", ".join(CATEGORIES)
    return f"""Target keyword: "{keyword}"

Page title: "{page_title}"

TASK: Extract all terms from this page that are important for writing content
about the target keyword. Replicate Google NLP API entity extraction:
- Include entities regardless of frequency — salience reflects semantic importance
- Include domain-specific concepts (health conditions, care routines) not just named entities
- Normalize variants: "Jack Russells" → "jack russell terrier"
- Include single-word trait adjectives (affectionate, intelligent, playful) when they
  describe the topic subject
- Include temporal phrases (15 years, golden years) as their own category
- Do NOT include: navigational text, author names, unrelated brand mentions, UI artifacts

Return JSON:
{{
  "terms": [
    {{
      "term": "exact normalized phrase",
      "category": "one of [{cats_str}]",
      "salience": 0.0-1.0,
      "normalized": "canonical form if different from term"
    }}
  ]
}}

PAGE TEXT (first 3000 chars):
{body_text[:3000]}"""


def build_gap_fill_prompt(keyword: str, extracted_terms: list[str]) -> str:
    cats_str = ", ".join(CATEGORIES)
    return f"""Target keyword: "{keyword}"

The following terms were already found in competitor pages:
{json.dumps(extracted_terms[:60])}

TASK: Add up to 25 important terms for this topic that are NOT in the list above.
These should be terms that expert content on this topic would typically include,
even if they didn't happen to appear in the specific pages analyzed.

Return JSON:
{{"generated": [{{"term": "...", "category": "one of [{cats_str}]", "salience": 0.0-1.0}}]}}"""


def _normalize_term_entry(entry: dict[str, Any]) -> dict[str, Any]:
    term = str(entry.get("term", "")).strip()
    norm = entry.get("normalized")
    if norm is not None and str(norm).strip():
        normalized = str(norm).strip()
    else:
        normalized = term
    try:
        salience = float(entry.get("salience", 0.5))
    except (TypeError, ValueError):
        salience = 0.5
    salience = max(0.0, min(1.0, salience))
    category = str(entry.get("category", "other")).strip() or "other"
    if category not in CATEGORIES:
        category = "other"
    return {
        "term": (normalized or term).lower().strip() or term.lower().strip(),
        "category": category,
        "salience": salience,
        "normalized": (normalized or term).lower().strip(),
    }


async def extract_terms_from_page_async(
    keyword: str,
    page: CleanDocument,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    try:
        raw = await _chat_completion(
            client,
            system=SYSTEM_PROMPT,
            user=build_extraction_prompt(keyword, page.title_text, page.body_text),
        )
        clean = _strip_json_fences(raw)
        parsed = json.loads(clean)
        terms = parsed.get("terms", [])
        if not isinstance(terms, list):
            return []
        out: list[dict[str, Any]] = []
        for t in terms:
            if isinstance(t, dict):
                out.append(_normalize_term_entry(t))
        return out
    except (httpx.TimeoutException, httpx.ConnectError):
        logger.debug("[llm_extractor] Skipped (timeout/connect) %s", page.url)
        return []
    except Exception as e:
        logger.warning(
            "[llm_extractor] Failed for %s error_type=%s error=%r",
            page.url,
            type(e).__name__,
            e,
        )
        return []


async def generate_gap_terms_async(
    keyword: str,
    extracted_terms: list[str],
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    try:
        raw = await _chat_completion(
            client,
            system="You are an SEO content strategist. Return only valid JSON.",
            user=build_gap_fill_prompt(keyword, extracted_terms),
        )
        clean = _strip_json_fences(raw)
        parsed = json.loads(clean)
        generated = parsed.get("generated", [])
        if not isinstance(generated, list):
            return []
        out: list[dict[str, Any]] = []
        for t in generated:
            if not isinstance(t, dict):
                continue
            base = _normalize_term_entry(t)
            sal = base["salience"]
            out.append(
                {
                    **base,
                    "source": "llm_generated",
                    "score": round(sal * 0.5, 4),
                }
            )
        return out
    except (httpx.TimeoutException, httpx.ConnectError):
        logger.debug("[llm_extractor] Gap-fill skipped (timeout/connect)")
        return []
    except Exception as e:
        logger.warning(
            "[llm_extractor] Gap-fill failed error_type=%s error=%r",
            type(e).__name__,
            e,
        )
        return []


async def _check_gamma_reachable() -> bool:
    """Quick probe to see if the gamma endpoint is reachable before bulk requests."""
    probe_timeout = httpx.Timeout(60.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=probe_timeout) as probe:
            resp = await probe.post(
                GAMMA_CHAT_URL,
                json={
                    "model": GAMMA_MODEL,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code < 500
    except Exception:
        return False


async def extract_all_pages_async(
    keyword: str,
    documents: list[CleanDocument],
) -> list[list[dict[str, Any]]]:
    """Run per-page extraction with bounded concurrency."""
    if not documents:
        return []

    reachable = await _check_gamma_reachable()
    if not reachable:
        logger.warning(
            "[llm_extractor] Gamma API unreachable — skipping LLM extraction for %d pages",
            len(documents),
        )
        return [[] for _ in documents]

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _guarded(doc: CleanDocument) -> list[dict[str, Any]]:
        async with semaphore:
            return await extract_terms_from_page_async(keyword, doc, client)

    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        tasks = [_guarded(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid: list[list[dict[str, Any]]] = []
    for i, r in enumerate(results):
        if isinstance(r, list):
            valid.append(r)
        elif isinstance(r, BaseException):
            logger.warning(
                "[llm_extractor] Page task failed for %s error_type=%s error=%r",
                documents[i].url if i < len(documents) else "?",
                type(r).__name__,
                r,
            )
            valid.append([])
        else:
            logger.warning("[llm_extractor] Unexpected result type: %s", type(r))
            valid.append([])
    return valid
