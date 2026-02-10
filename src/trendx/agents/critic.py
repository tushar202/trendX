from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Heuristic "Critic" / Scorer
# NOTE: This module is currently a "Repo Quality Scorer" rather than a true
# semantic critic. It calculates a 'production_score' based on GitHub metadata.
#
# DISTINCTION: This is distinct from the LLM-based "Critic" used inside
# `src/trendx/agents/synthesis.py`, which evaluates the quality of the written report.
# -----------------------------------------------------------------------------


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
    """
    Calculate a heuristic score (0-100) for repository maturity.
    
    Factors:
    +20: Docker support (Dockerfile or compose)
    +20: Tests (has_tests flag)
    +20: Permissive License (MIT/Apache 2.0)
    -30: "Hype" Penalty (High stars > 1000 but very young < 30 days)
    """
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
    """
    Run the heuristic critic on the provided trends.
    Computes and aggregates 'production_score' for items in each trend.
    """
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
