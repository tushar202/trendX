from __future__ import annotations

import re
from typing import Any, Dict, List


def _evidence_strength(item: Dict[str, Any]) -> float:
    score = 0.4
    text = (item.get("cleaned_text") or item.get("raw_text") or "").lower()
    if re.search(r"benchmark|evaluation|results|ablation", text):
        score += 0.2
    meta = item.get("repo_metadata") or item.get("github_metadata") or {}
    if meta.get("has_tests") or meta.get("has_tests_folder"):
        score += 0.1
    if meta.get("has_dockerfile") or meta.get("has_docker_compose"):
        score += 0.1
    if meta.get("license") in ["MIT", "Apache-2.0"]:
        score += 0.1
    return min(1.0, score)


def apply_trust_scores(items: List[Dict[str, Any]], source_weights: Dict[str, float]) -> None:
    for item in items:
        source = item.get("source") or "unknown"
        base = float(source_weights.get(source, 0.5))
        evidence = _evidence_strength(item)
        contradiction_penalty = 0.1 if item.get("conflict") else 0.0
        corroboration = float(item.get("corroboration_factor", 1.0))
        trust = base * evidence * corroboration - contradiction_penalty
        item["trust_score"] = max(0.0, min(1.0, trust))
