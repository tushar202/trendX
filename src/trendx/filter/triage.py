from __future__ import annotations

from typing import Any, Dict, List


def filter_items(
    items: List[Dict[str, Any]],
    min_text_len: int = 200,
    min_title_len: int = 5,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for item in items:
        title = (item.get("title") or "").strip()
        text = (item.get("cleaned_text") or item.get("raw_text") or "").strip()
        if len(title) < min_title_len and len(text) < min_text_len:
            continue
        kept.append(item)
    return kept
