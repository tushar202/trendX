from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from trendx.utils.simhash import simhash


def _item_text(item: Dict[str, Any]) -> str:
    return (
        item.get("cleaned_text")
        or item.get("raw_text")
        or item.get("title")
        or item.get("url")
        or ""
    )


def _ensure_embeddings(items: List[Dict[str, Any]], model_name: str) -> None:
    missing = [i for i in items if not i.get("embedding")]
    if not missing:
        return
    model = SentenceTransformer(model_name)
    texts = [_item_text(i) for i in missing]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    for item, vec in zip(missing, vectors):
        item["embedding"] = vec


def dedupe_items(
    items: List[Dict[str, Any]],
    embedding_model: str,
    sim_threshold: float = 0.92,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    _ensure_embeddings(items, embedding_model)

    kept: List[Dict[str, Any]] = []
    kept_vecs: List[np.ndarray] = []

    for item in items:
        canon = item.get("canonical_url") or item.get("url")
        text = _item_text(item)
        item_hash = simhash(text)

        duplicate_idx = None

        for idx, kept_item in enumerate(kept):
            kept_canon = kept_item.get("canonical_url") or kept_item.get("url")
            if canon and kept_canon and canon == kept_canon:
                duplicate_idx = idx
                break
            kept_hash = kept_item.get("simhash")
            if kept_hash is not None and kept_hash == item_hash:
                duplicate_idx = idx
                break

            if item.get("embedding") and kept_item.get("embedding"):
                v = np.array(item["embedding"], dtype=float)
                kv = kept_vecs[idx]
                sim = float(v @ kv)
                if sim >= sim_threshold:
                    duplicate_idx = idx
                    break

        if duplicate_idx is None:
            item["simhash"] = item_hash
            kept.append(item)
            kept_vecs.append(np.array(item.get("embedding") or [], dtype=float))
            continue

        existing = kept[duplicate_idx]
        existing_score = existing.get("trust_score") or 0.0
        new_score = item.get("trust_score") or 0.0
        if new_score > existing_score:
            item["simhash"] = item_hash
            kept[duplicate_idx] = item
            kept_vecs[duplicate_idx] = np.array(item.get("embedding") or [], dtype=float)

    return kept
