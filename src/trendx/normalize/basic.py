from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        url = (item.get("url") or "").strip()
        canonical_url = _canonicalize_url(url) if url else None
        text = (item.get("raw_text") or "").strip()
        normalized.append(
            {
                **item,
                "canonical_url": canonical_url or item.get("url"),
                "cleaned_text": " ".join(text.split()),
            }
        )
    return normalized


def _canonicalize_url(url: str) -> str:
    parts = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")]
    query = [(k, v) for k, v in query if k not in {"ref", "source", "fbclid", "gclid"}]
    cleaned = parts._replace(
        scheme=parts.scheme.lower(),
        netloc=parts.netloc.lower(),
        query=urlencode(query),
        fragment="",
    )
    return urlunparse(cleaned)
