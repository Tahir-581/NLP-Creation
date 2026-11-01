"""HTTP client for GPT-OSS-120B (same contract as GPT-OSS-120B-PIPELINE/llm_extractor.py)."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GAMMA_CHAT_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
GAMMA_MODEL = os.environ.get("LLM_MODEL", "gptoss120")
GAMMA_API_KEY = ""  # set LLM_API_KEY in the environment

DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=8.0)


def _gamma_api_key() -> str:
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or GAMMA_API_KEY
    ).strip()


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def assistant_text_from_response(data: Any) -> str:
    """Parse assistant text from OpenAI-style chat completion JSON."""
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


async def chat_completion(
    client: httpx.AsyncClient,
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """POST to GPT-OSS-120B; returns raw assistant text (expect JSON inside)."""
    payload: dict[str, Any] = {
        "model": GAMMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
            "[gpt_oss_client] HTTP error url=%s status=%s error_type=%s body=%r",
            GAMMA_CHAT_URL,
            status,
            type(exc).__name__,
            body,
        )
        raise
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        logger.debug(
            "[gpt_oss_client] Request error url=%s error_type=%s error=%r",
            GAMMA_CHAT_URL,
            type(exc).__name__,
            exc,
        )
        raise
    except httpx.RequestError as exc:
        logger.warning(
            "[gpt_oss_client] Request error url=%s error_type=%s error=%r",
            GAMMA_CHAT_URL,
            type(exc).__name__,
            exc,
        )
        raise

    try:
        data = resp.json()
    except ValueError as exc:
        logger.warning(
            "[gpt_oss_client] Invalid JSON status=%s body=%r error=%r",
            resp.status_code,
            _truncate(resp.text.strip() if isinstance(resp.text, str) else ""),
            exc,
            exc_info=True,
        )
        raise

    text = assistant_text_from_response(data)
    return text if isinstance(text, str) else str(text or "")
