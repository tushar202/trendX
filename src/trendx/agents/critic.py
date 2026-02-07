from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _repo_age_days(meta: Dict[str, Any]) -> Optional[int]:
    raw = meta.get("repo_age_days")
    if isinstance(raw, int):
        return raw
    created_at = _parse_dt(meta.get("created_at"))
    if not created_at:
        return None
    now = datetime.now(tz=timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max(0, (now - created_at).days)


def _production_score(meta: Dict[str, Any]) -> int:
    score = 0
    if meta.get("has_dockerfile") or meta.get("has_docker_compose"):
        score += 20
    if meta.get("has_tests") or meta.get("has_tests_folder"):
        score += 20
    if meta.get("license") in ["MIT", "Apache-2.0"]:
        score += 20

    stars = int(meta.get("stars") or 0)
    age_days = _repo_age_days(meta)
    if age_days is not None and age_days < 30 and stars > 1000:
        score -= 30

    return score


def run(trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for trend in trends:
        scores: List[int] = []
        for item in trend.get("items", []):
            meta = item.get("repo_metadata") or item.get("repo") or {}
            if not isinstance(meta, dict) or not meta:
                continue
            score = _production_score(meta)
            item["production_score"] = score
            scores.append(score)
        trend["production_score"] = sum(scores) / len(scores) if scores else None
    return trends
