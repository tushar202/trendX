from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def search_cluster_evidence(
    items: List[Dict[str, Any]],
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 2,
    max_chars: int = 400,
) -> str:
    valid_items = [i for i in items if i.get("embedding") is not None]
    if not valid_items or not query or not query.strip():
        return "No searchable evidence found."

    model = get_embedding_model(model_name)
    query_vec = model.encode(query, normalize_embeddings=True)

    results = []
    for item in valid_items:
        item_vec = np.array(item.get("embedding"), dtype=float)
        if item_vec.shape != query_vec.shape:
            continue
        score = float(np.dot(query_vec, item_vec))
        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)

    output_lines = []
    total_len = 0
    for score, item in results[:top_k]:
        if score < 0.25:
            continue
        text_content = item.get("cleaned_text") or item.get("raw_text") or ""
        snippet = text_content[:max_chars].replace("\n", " ").strip()
        entry = (
            f"- [Rel: {score:.2f}] {item.get('title', 'Source')}: \"{snippet}...\""
        )
        if total_len + len(entry) > 1000:
            break
        output_lines.append(entry)
        total_len += len(entry)

    return "\n".join(output_lines) if output_lines else "No highly relevant evidence found."
