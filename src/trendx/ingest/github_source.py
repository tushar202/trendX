from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from trendx.utils.logging import get_logger

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"


def _headers(token_env: Optional[str]) -> Dict[str, str]:
    token = os.getenv(token_env) if token_env else None
    if not token:
        return {"Accept": "application/vnd.github+json"}
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def fetch_github_repos(
    query: str, max_results: int = 50, token_env: Optional[str] = "GITHUB_TOKEN"
) -> List[Dict[str, Any]]:
    headers = _headers(token_env)
    per_page = min(100, max_results)
    page = 1
    items: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30) as client:
        while len(items) < max_results:
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }
            try:
                resp = await client.get(
                    f"{GITHUB_API}/search/repositories", params=params, headers=headers
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403, 429):
                    logger.warning(
                        "GitHub search failed (status %s). Set GITHUB_TOKEN or lower limits.",
                        status,
                    )
                    return items
                raise
            data = resp.json()
            repos = data.get("items", [])
            if not repos:
                break

            for repo in repos:
                items.append(
                    {
                        "source": "github",
                        "url": repo.get("html_url"),
                        "repo_url": repo.get("html_url"),
                        "title": repo.get("full_name"),
                        "authors": (repo.get("owner") or {}).get("login"),
                        "published_at": repo.get("created_at"),
                        "raw_text": repo.get("description") or "",
                        "github_metadata": {
                            "stars": repo.get("stargazers_count"),
                            "forks": repo.get("forks_count"),
                            "license": (repo.get("license") or {}).get("spdx_id"),
                        },
                    }
                )
                if len(items) >= max_results:
                    break

            page += 1

    return items
