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


async def tavily_search(
    query: str,
    api_key: str | None = None,
    max_results: int = 3,
) -> str:
    """Perform an external web search using Tavily."""
    if not api_key:
        return "External search unavailable: No TAVILY_API_KEY provided."

    try:
        # Try using the official client first
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            resp = client.search(query, max_results=max_results)
            results = resp.get("results", [])
        except ImportError:
            # Fallback to direct HTTP request if library is missing
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                }
                async with session.post("https://api.tavily.com/search", json=payload) as r:
                    if r.status != 200:
                        return f"Search failed with status {r.status}."
                    data = await r.json()
                    results = data.get("results", [])

        if not results:
            return "No external results found."

        output = []
        for res in results:
            title = res.get("title", "Untitled")
            url = res.get("url", "#")
            content = res.get("content", "")[:300].replace("\n", " ").strip()
            output.append(f"- [WEB] {title} ({url}): \"{content}...\"")

        return "\n".join(output)

    except Exception as e:
        return f"External search error: {str(e)}"
