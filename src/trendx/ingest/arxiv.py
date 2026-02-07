from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"


def _parse_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return value


async def fetch_arxiv_items(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(ARXIV_API, params=params)
        resp.raise_for_status()
        xml_text = resp.text

    # Lightweight Atom parsing without extra deps
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    items: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default=None, namespaces=ns)
        url = entry.findtext("atom:id", default="", namespaces=ns)

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(name.strip())

        items.append(
            {
                "source": "arxiv",
                "url": url.strip(),
                "title": title,
                "authors": ", ".join(authors),
                "published_at": _parse_datetime(published),
                "raw_text": summary,
            }
        )

    return items
