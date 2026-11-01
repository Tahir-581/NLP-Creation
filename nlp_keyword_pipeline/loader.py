"""Load JSON corpus from a directory or ZIP archive."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

try:
    import orjson

    def _loads(b: bytes | str) -> Any:
        if isinstance(b, str):
            return orjson.loads(b.encode())
        return orjson.loads(b)

except ImportError:
    _loads = lambda b: json.loads(b if isinstance(b, str) else b.decode("utf-8"))

from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.models import PageDocument

logger = logging.getLogger(__name__)

REQUIRED_MIN_TEXT_FIELDS = ("title", "headings", "paragraphs", "content")


def _usable_text_field_names(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if str(payload.get("title", "") or "").strip():
        out.append("title")
    hs = payload.get("headings")
    if hs and any(str(h or "").strip() for h in hs):
        out.append("headings")
    ps = payload.get("paragraphs")
    if ps and any(str(p or "").strip() for p in ps):
        out.append("paragraphs")
    if str(payload.get("content", "") or "").strip():
        out.append("content")
    return out


def _detect_boilerplate_ratio(payload: dict[str, Any], config: PipelineConfig) -> float:
    """Approximate share of lines that look like boilerplate / promo."""
    chunks: list[str] = []
    for key in ("title", "content"):
        v = payload.get(key)
        if v:
            chunks.append(str(v))
    hs = payload.get("headings")
    if hs:
        chunks.extend(str(h) for h in hs if h)
    ps = payload.get("paragraphs")
    if ps:
        chunks.extend(str(p) for p in ps if p)
    raw = "\n".join(chunks)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    junk_hits = 0
    for ln in lines:
        low = ln.lower()
        if any(pat in low for pat in config.junk_phrases):
            junk_hits += 1
            continue
        if any(pat in low for pat in config.blocked_title_patterns):
            junk_hits += 1
    return min(1.0, junk_hits / len(lines))


def extract_domain(url: str) -> str:
    if not url:
        return ""
    domain = url.replace("https://", "").replace("http://", "").split("/")[0].lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def is_payload_usable(payload: dict[str, Any]) -> bool:
    if "error" in payload and len(payload.keys()) <= 2:
        return False
    if not payload.get("url"):
        return False
    return any(payload.get(f) for f in REQUIRED_MIN_TEXT_FIELDS)


def to_page_document(
    path: Path,
    payload: dict[str, Any],
    config: PipelineConfig | None = None,
) -> PageDocument:
    doc_id = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]
    url = str(payload.get("url", "") or "")
    domain = extract_domain(url)
    headings = payload.get("headings")
    if headings is None:
        headings = []
    paragraphs = payload.get("paragraphs")
    if paragraphs is None:
        paragraphs = []
    title_str = str(payload.get("title", "") or "")
    title_lower = title_str.lower()
    cfg = config or PipelineConfig()
    has_error = any(pat in title_lower for pat in cfg.blocked_title_patterns)
    usable = _usable_text_field_names(payload)
    para_list = [str(p) for p in paragraphs if p is not None]
    bp_ratio = _detect_boilerplate_ratio(payload, cfg)
    return PageDocument(
        doc_id=doc_id,
        file_path=str(path),
        url=url,
        domain=domain,
        title=title_str,
        description=str(payload.get("description", "") or ""),
        headings=[str(h) for h in headings if h is not None],
        paragraphs=para_list,
        content=str(payload.get("content", "") or ""),
        word_count=payload.get("word_count"),
        heading_count=payload.get("heading_count"),
        para_count=payload.get("para_count"),
        authority=payload.get("authority"),
        has_error_signals=has_error,
        usable_text_fields=usable,
        paragraph_count=len(para_list),
        boilerplate_ratio=bp_ratio,
    )


def unzip_dataset(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return list(out_dir.rglob("*.json"))


def discover_json_paths(source: Path, config: PipelineConfig) -> list[Path]:
    """If source is a file ending in .zip, unzip to work_dir and return json paths."""
    source = source.resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        out = Path(config.work_dir) / "unzipped"
        return unzip_dataset(source, out)
    if source.is_dir():
        return sorted(source.rglob("*.json"))
    raise ValueError(f"Source must be a directory or .zip file: {source}")


def parse_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_bytes()
        payload = _loads(raw)
        if not isinstance(payload, dict):
            return None, "root_not_object"
        return payload, None
    except Exception as exc:
        logger.warning("Parse failed %s: %s", path, exc)
        return None, str(exc)


def load_corpus(
    source: str | Path,
    config: PipelineConfig,
) -> tuple[list[PageDocument], list[dict[str, Any]]]:
    """
    Load all valid page documents from a folder or ZIP.

    Returns (documents, parse_issues).
    """
    paths = discover_json_paths(Path(source), config)
    docs: list[PageDocument] = []
    issues: list[dict[str, Any]] = []

    for path in paths:
        payload, err = parse_json_file(path)
        if err:
            issues.append({"file": str(path), "stage": "parse", "error": err})
            continue
        assert payload is not None
        if not is_payload_usable(payload):
            issues.append(
                {
                    "file": str(path),
                    "stage": "validate",
                    "error": "No usable text fields or error-only payload",
                }
            )
            continue
        docs.append(to_page_document(path, payload, config))

    logger.info("Loaded %d valid documents, %d issues", len(docs), len(issues))
    return docs, issues
