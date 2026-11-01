"""Async LLM keyword extraction via an OpenAI-compatible endpoint (GPT-OSS-120B)."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, cast

import httpx

from heading_outline import (
    apply_llm_assignments,
    fallback_all_h2_outline,
    hierarchy_to_outline,
    needs_outline_level_llm,
    ordered_unique_headings,
)
from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)

GAMMA_CHAT_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
GAMMA_MODEL = os.environ.get("LLM_MODEL", "gptoss120")
GAMMA_API_KEY = ""  # set LLM_API_KEY in the environment

# Keep read timeout generous because GPT-OSS-120B can exceed 25s per page.
_LLM_TIMEOUT = httpx.Timeout(90.0, connect=8.0)
# Keep concurrency modest to reduce queueing/timeouts on the hosted endpoint.
_CONCURRENCY = 2

SYSTEM_PROMPT_HEADING_LEVELS = """You assign outline levels only. You must copy heading text exactly
as given — no paraphrasing, no new lines.

JSON rules (mandatory):
1. Output exactly one JSON object — no markdown fences, no text before or after.
2. Use double quotes for all keys and string values; no single quotes, no trailing commas.
3. If a heading contains a double quote character, escape it as \\" inside JSON strings."""

SYSTEM_PROMPT = """You are an SEO content analyst replicating Google Natural Language API behavior.
Given a competitor page and a target keyword, extract all terms a content writer would need
to achieve topical coverage.

JSON rules (mandatory):
1. Output exactly one JSON object — no markdown, no explanation, no text outside JSON.
2. Use double quotes for all keys and string values; no trailing commas after the last item.
3. Escape any double quote inside a string value as \\" (e.g. breed nicknames or copied titles).
4. Do not truncate the response mid-string; shorten the terms list if needed to stay valid."""

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

# Logs live at repo root / logs (parent of GPT-OSS-120B-PIPELINE).
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Default token budget for term extraction; retry may raise further to reduce truncation.
_DEFAULT_EXTRACTION_MAX_TOKENS = 1800
_RETRY_MIN_TOKENS = 2200

RETRY_USER_SUFFIX = """

IMPORTANT: Your previous reply was not valid complete JSON. Reply again with ONE compact JSON object only:
- Double-quoted keys and strings; no trailing commas; no markdown or prose.
- Escape any " inside string values as \\".
- Do not truncate strings mid-value; omit lower-priority terms if needed to finish valid JSON."""


def _gamma_api_key() -> str:
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or GAMMA_API_KEY
    ).strip()


def _strip_json_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()


def _clean_json_text(raw: str) -> str:
    """Strip fences, BOM, whitespace; remove trailing commas before } or ]."""
    text = _strip_json_fences(raw or "")
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.strip()
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _extract_first_json_candidate(cleaned: str) -> str:
    """Return the first JSON object/array substring using raw_decode when possible."""
    text = cleaned
    if not text:
        return ""
    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        try:
            _, end_idx = decoder.raw_decode(text[start_idx:])
            return text[start_idx : start_idx + end_idx]
        except json.JSONDecodeError:
            continue
    return text


def _json_looks_truncated(candidate: str) -> bool:
    """Cheap heuristic: unbalanced delimiters or abrupt end."""
    s = candidate.strip()
    if not s:
        return True
    if s[0] == "{":
        if s.count("{") > s.count("}"):
            return True
        if not s.endswith("}"):
            return True
    elif s[0] == "[":
        if s.count("[") > s.count("]"):
            return True
        if not s.endswith("]"):
            return True
    return False


def _parse_llm_json(raw: str) -> tuple[Any | None, bool, str | None]:
    """
    Parse model JSON. Returns (parsed, truncated_guess, error_detail).
    Does not log long payloads — callers write debug files on final failure.
    """
    cleaned = _clean_json_text(raw)
    candidate = _extract_first_json_candidate(cleaned)
    if not candidate:
        return None, True, "empty response"
    truncated_guess = _json_looks_truncated(candidate)
    try:
        return json.loads(candidate), truncated_guess, None
    except json.JSONDecodeError as exc:
        err = str(exc)
        truncated_guess = truncated_guess or "unterminated" in err.lower()
        return None, truncated_guess, err


def _save_llm_json_debug(context: str, raw_text: str, detail: str) -> Path:
    """Write raw LLM output when JSON parsing failed after retry. Logs dir: repo root / logs."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\-.]+", "_", context).strip("_")[:80] or "ctx"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _LOGS_DIR / f"llm_json_fail_{slug}_{ts}.txt"
    path.write_text(
        f"context: {context}\nerror: {detail}\n\n--- raw ---\n{raw_text}",
        encoding="utf-8",
    )
    logger.warning(
        "[llm_extractor] JSON parse failed context=%s — see %s",
        context,
        path,
    )
    return path


async def _chat_completion_parse_json(
    client: httpx.AsyncClient,
    *,
    system: str,
    user: str,
    context: str,
    max_tokens: int = _DEFAULT_EXTRACTION_MAX_TOKENS,
    temperature: float = 0.3,
) -> tuple[Any | None, str]:
    """
    One LLM call plus one retry on parse failure. Returns (parsed, raw_text for salvage).
    """
    raw = await _chat_completion(
        client,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed, _, err = _parse_llm_json(raw)
    if parsed is not None:
        return parsed, raw

    retry_tokens = max(max_tokens, _RETRY_MIN_TOKENS)
    raw2 = await _chat_completion(
        client,
        system=system,
        user=user + RETRY_USER_SUFFIX,
        max_tokens=retry_tokens,
        temperature=temperature,
    )
    parsed2, _, err2 = _parse_llm_json(raw2)
    if parsed2 is not None:
        return parsed2, raw2

    detail = f"after retry: {err2 or err or 'parse failed'}"
    _save_llm_json_debug(context, raw2, detail)
    return None, raw2


def _extract_term_list(parsed: Any, *, keys: tuple[str, ...]) -> list[Any]:
    """Extract term rows from multiple possible response shapes."""
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []

    for key in keys:
        val = parsed.get(key)
        if isinstance(val, list):
            return val

    for wrapper in ("data", "result", "output", "payload"):
        nested = parsed.get(wrapper)
        if isinstance(nested, dict):
            for key in keys:
                val = nested.get(key)
                if isinstance(val, list):
                    return val
    return []


def _salvage_term_objects(raw: str) -> list[dict[str, Any]]:
    """
    Best-effort recovery for malformed JSON: collect object literals containing "term".
    This intentionally accepts partial responses so a single malformed token
    doesn't wipe all extracted terms.
    """
    text = _clean_json_text(raw or "")
    if not text:
        return []

    recovered: list[dict[str, Any]] = []
    for match in re.finditer(r"\{[^{}]*\"term\"\s*:\s*\"[^\"\\]*(?:\\.[^\"\\]*)*\"[^{}]*\}", text, flags=re.DOTALL):
        obj_text = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            parsed = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("term"):
            recovered.append(parsed)
    return recovered


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
                # Some providers emit only a reasoning text field on partial outputs.
                for key in ("reasoning", "reasoning_content"):
                    reasoning = msg.get(key)
                    if isinstance(reasoning, str) and reasoning.strip():
                        return reasoning

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
    max_tokens: int = _DEFAULT_EXTRACTION_MAX_TOKENS,
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
        # Improve structured output reliability on reasoning-capable models.
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
    }
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_gamma_api_key()}",
    }

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

JSON rules for your reply:
1. One JSON object only; no markdown fences, no commentary.
2. Double-quoted keys and strings; no trailing commas.
3. Escape any " inside string values as \\".

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


def build_heading_level_prompt(ordered_headings: list[str]) -> str:
    """Ask for one level per heading; strings must be echoed verbatim in the JSON."""
    payload = json.dumps(ordered_headings, ensure_ascii=False)
    return f"""These are section headings from a single web page, in order.

TASK: Assign each heading exactly one level: 1 (main title / top section), 2 (subsection), or 3 (deeper subsection).
Use only the strings below — copy each "text" field exactly (same spelling, spacing, and punctuation).

Headings (JSON array):
{payload}

JSON rules for your reply:
1. One JSON object only; no markdown, no extra text.
2. Double-quoted keys and strings; escape " in heading text as \\".
3. No trailing commas.

Return JSON:
{{
  "assignments": [
    {{ "text": "<exact string from the list above>", "level": 1 }},
    ...
  ]
}}

Rules:
- Include every heading from the array exactly once.
- "level" must be 1, 2, or 3.
- Do not add headings. Do not remove headings. Do not edit heading text."""


def build_gap_fill_prompt(keyword: str, extracted_terms: list[str]) -> str:
    return f"""Target keyword: "{keyword}"

Top corpus-derived terms (already extracted from pages):
{json.dumps(extracted_terms[:60])}

TASK: Suggest up to 25 ADDITIONAL keyword terms that are tightly related to this topic
and could be implied or supported by expert coverage. Do NOT repeat any term from the list above.
Do NOT use a fixed category taxonomy — use short free-form "likely_category" labels only.

For each term explain briefly why it is supported (why_supported) and which signals would
support it (supporting_signals: e.g. heading, entity, title, stats).

JSON rules for your reply:
1. One JSON object only; no markdown or prose outside JSON.
2. Double-quoted keys and strings; no trailing commas; escape " in values as \\".

Return JSON:
{{"generated": [{{"term": "...", "likely_category": "...", "why_supported": "...", "supporting_signals": ["heading"], "salience": 0.0}}]}}"""


GAP_FILL_SYSTEM = """You are an SEO content strategist.

JSON rules (mandatory):
1. Output exactly one JSON object — no markdown, no text before or after.
2. Double-quoted keys and strings; no trailing commas.
3. Escape any double quote inside a string value as \\"."""


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
        parsed, raw = await _chat_completion_parse_json(
            client,
            system=SYSTEM_PROMPT,
            user=build_extraction_prompt(keyword, page.title_text, page.body_text),
            context=f"page:{page.url}",
        )
        terms: list[Any] = []
        if parsed is not None:
            terms = _extract_term_list(parsed, keys=("terms", "entities"))
        if not terms:
            salvaged = _salvage_term_objects(raw)
            if salvaged:
                logger.warning(
                    "[llm_extractor] Recovered %d term objects from malformed JSON page=%s",
                    len(salvaged),
                    page.url,
                )
                terms = salvaged
        out: list[dict[str, Any]] = []
        for t in terms:
            if isinstance(t, dict):
                out.append(_normalize_term_entry(t))
        return out
    except (httpx.TimeoutException, httpx.ConnectError):
        logger.warning(
            "[llm_extractor] Skipped (timeout/connect) page=%s timeout=%s",
            page.url,
            _LLM_TIMEOUT,
        )
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
        if not extracted_terms:
            logger.warning(
                "[llm_extractor] Gap-fill skipped: no corpus terms available from extraction"
            )
            return []
        parsed, raw = await _chat_completion_parse_json(
            client,
            system=GAP_FILL_SYSTEM,
            user=build_gap_fill_prompt(keyword, extracted_terms),
            context=f"gap:{keyword}",
            max_tokens=1600,
        )
        generated: list[Any] = []
        if parsed is not None:
            generated = _extract_term_list(parsed, keys=("generated", "terms", "suggestions"))
        if not generated:
            salvaged = _salvage_term_objects(raw)
            if salvaged:
                logger.warning(
                    "[llm_extractor] Recovered %d gap-fill terms from malformed JSON",
                    len(salvaged),
                )
                generated = salvaged
        out: list[dict[str, Any]] = []
        for t in generated:
            if not isinstance(t, dict):
                continue
            base = _normalize_term_entry(t)
            sal = base["salience"]
            likely = str(
                t.get("likely_category") or t.get("category") or base.get("category") or "other"
            ).strip()
            why = str(t.get("why_supported") or "").strip()
            sigs = t.get("supporting_signals")
            if isinstance(sigs, str):
                supporting = [sigs]
            elif isinstance(sigs, list):
                supporting = [str(x) for x in sigs if x]
            else:
                supporting = []
            out.append(
                {
                    **base,
                    "likely_category": likely,
                    "why_supported": why,
                    "supporting_signals": supporting,
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


async def assign_outline_levels_async(
    ordered_headings: list[str],
    client: httpx.AsyncClient,
    *,
    page_url: str = "",
) -> dict[str, list[str]]:
    """LLM assigns 1/2/3 per verbatim heading; on validation failure, all-h2 fallback."""
    if not ordered_headings:
        return {"h1": [], "h2": [], "h3": []}
    try:
        parsed, raw = await _chat_completion_parse_json(
            client,
            system=SYSTEM_PROMPT_HEADING_LEVELS,
            user=build_heading_level_prompt(ordered_headings),
            context=f"outline_levels:{page_url}",
            max_tokens=1400,
            temperature=0.1,
        )
        assignments: list[Any] = []
        if isinstance(parsed, dict):
            val = parsed.get("assignments")
            if isinstance(val, list):
                assignments = val
        result = apply_llm_assignments(ordered_headings, assignments)
        if result is None:
            logger.warning(
                "[llm_extractor] Outline level validation failed; using all-h2 fallback page=%s",
                page_url or "?",
            )
            return fallback_all_h2_outline(ordered_headings)
        return result
    except (httpx.TimeoutException, httpx.ConnectError):
        logger.warning(
            "[llm_extractor] Outline levels skipped (timeout/connect) page=%s",
            page_url or "?",
        )
        return fallback_all_h2_outline(ordered_headings)
    except Exception as e:
        logger.warning(
            "[llm_extractor] Outline levels failed page=%s error_type=%s error=%r",
            page_url or "?",
            type(e).__name__,
            e,
        )
        return fallback_all_h2_outline(ordered_headings)


async def extract_page_heading_outlines_async(
    documents: list[CleanDocument],
    *,
    outline_llm_levels: bool,
    config: PipelineConfig | None = None,
) -> list[dict[str, list[str]]]:
    """
    One outline dict per document (h1/h2/h3). Uses LLM only when all filtered
    headings are crawl-default level 2 and outline_llm_levels is True.
    """
    cfg = config or PipelineConfig()
    if not documents:
        return []

    n = len(documents)
    out: list[dict[str, list[str]] | None] = [None] * n
    need_llm_indices: list[int] = []

    for i, doc in enumerate(documents):
        if outline_llm_levels and needs_outline_level_llm(
            doc.heading_hierarchy, config=cfg
        ):
            need_llm_indices.append(i)
        else:
            out[i] = hierarchy_to_outline(doc.heading_hierarchy, config=cfg)

    if not need_llm_indices:
        return cast(list[dict[str, list[str]]], out)

    reachable = await _check_gamma_reachable()
    if not reachable:
        logger.warning(
            "[llm_extractor] Gamma unreachable — outline LLM skipped for %d pages",
            len(need_llm_indices),
        )
        for i in need_llm_indices:
            doc = documents[i]
            ordered = ordered_unique_headings(doc.heading_hierarchy, config=cfg)
            out[i] = fallback_all_h2_outline(ordered)
        return cast(list[dict[str, list[str]]], out)

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _one(idx: int) -> dict[str, list[str]]:
        doc = documents[idx]
        ordered = ordered_unique_headings(doc.heading_hierarchy, config=cfg)
        async with semaphore:
            return await assign_outline_levels_async(
                ordered,
                client,
                page_url=doc.url,
            )

    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        tasks = [_one(i) for i in need_llm_indices]
        llm_results = await asyncio.gather(*tasks, return_exceptions=True)

    for j, idx in enumerate(need_llm_indices):
        r = llm_results[j]
        doc = documents[idx]
        ordered = ordered_unique_headings(doc.heading_hierarchy, config=cfg)
        if isinstance(r, dict):
            out[idx] = r
        else:
            if isinstance(r, BaseException):
                logger.warning(
                    "[llm_extractor] Outline task failed for %s error_type=%s error=%r",
                    doc.url,
                    type(r).__name__,
                    r,
                )
            out[idx] = fallback_all_h2_outline(ordered)

    return cast(list[dict[str, list[str]]], out)


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
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {_gamma_api_key()}",
                },
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
