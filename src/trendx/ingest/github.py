from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

GITHUB_API = "https://api.github.com"


def _extract_repo_full_name(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if url.startswith("git@github.com:"):
        path = url.split(":", 1)[1]
    elif "github.com" in url:
        parts = url.split("github.com", 1)[1]
        path = parts.strip("/")
    else:
        return None

    # strip .git and trailing paths
    path = path.replace(".git", "")
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2:
        return None
    return f"{segments[0]}/{segments[1]}"


def _headers(token_env: Optional[str]) -> Dict[str, str]:
    token = os.getenv(token_env) if token_env else None
    if not token:
        return {"Accept": "application/vnd.github+json"}
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _exists(client: httpx.AsyncClient, url: str, headers: Dict[str, str]) -> bool:
    resp = await client.get(url, headers=headers)
    if resp.status_code == 200:
        return True
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return False


async def fetch_repo_metadata(
    repo_full_name: str, token_env: Optional[str] = "GITHUB_TOKEN"
) -> Dict[str, Any]:
    headers = _headers(token_env)
    async with httpx.AsyncClient(timeout=30) as client:
        repo_url = f"{GITHUB_API}/repos/{repo_full_name}"
        try:
            resp = await client.get(repo_url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403, 429):
                return {}
            raise
        repo = resp.json()

        default_branch = repo.get("default_branch", "main")
        base = f"{GITHUB_API}/repos/{repo_full_name}/contents"

        has_dockerfile = await _exists(
            client, f"{base}/Dockerfile?ref={default_branch}", headers
        )
        has_compose = await _exists(
            client, f"{base}/docker-compose.yml?ref={default_branch}", headers
        ) or await _exists(
            client, f"{base}/docker-compose.yaml?ref={default_branch}", headers
        )
        has_tests = await _exists(client, f"{base}/tests?ref={default_branch}", headers)

        return {
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "created_at": repo.get("created_at"),
            "updated_at": repo.get("updated_at"),
            "license": (repo.get("license") or {}).get("spdx_id"),
            "default_branch": default_branch,
            "has_tests": bool(has_tests),
            "has_dockerfile": bool(has_dockerfile),
            "has_docker_compose": bool(has_compose),
        }


def repo_full_name_from_item(item: Dict[str, Any]) -> Optional[str]:
    url = (
        item.get("repo_url")
        or item.get("github_url")
        or (item.get("url") if item.get("source") == "github" else None)
    )
    if not url:
        return None
    return _extract_repo_full_name(url)
