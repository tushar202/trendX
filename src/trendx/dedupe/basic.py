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


def _merge_items(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Merge metadata from source item into target item."""
    # 1. Merge duplicate counts
    target_count = target.get("duplicate_count", 1)
    source_count = source.get("duplicate_count", 1)
    target["duplicate_count"] = target_count + source_count

    # 2. Merge URLs
    if "related_urls" not in target:
        target["related_urls"] = []
    # Add source's main URL
    if source.get("url") and source["url"] not in target["related_urls"] and source["url"] != target.get("url"):
        target["related_urls"].append(source["url"])
    # Add source's existing related_urls
    if source.get("related_urls"):
        for url in source["related_urls"]:
            if url not in target["related_urls"] and url != target.get("url"):
                target["related_urls"].append(url)

    # 3. Merge Claims/Evidence
    if "claims" not in target:
        target["claims"] = []
    if source.get("claims"):
        # Simple append for now; could dedupe claims in future if needed
        target["claims"].extend(source["claims"])


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
        # Initialize defaults for new items
        if "duplicate_count" not in item:
            item["duplicate_count"] = 1
        if "related_urls" not in item:
            item["related_urls"] = []
        if "claims" not in item:
            item["claims"] = []

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

        # Found a duplicate! Smart merge.
        existing = kept[duplicate_idx]
        existing_score = existing.get("trust_score") or 0.0
        new_score = item.get("trust_score") or 0.0

        if new_score > existing_score:
            # START MERGE: New item is better
            _merge_items(item, existing)
            item["simhash"] = item_hash
            kept[duplicate_idx] = item
            kept_vecs[duplicate_idx] = np.array(item.get("embedding") or [], dtype=float)
        else:
            # START MERGE: Existing item is better
            _merge_items(existing, item)
            # existing is updated in-place per reference logic

    return kept
