from __future__ import annotations

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FACT_KEYWORDS = [
    "we propose",
    "we introduce",
    "we present",
    "outperforms",
    "achieves",
    "state-of-the-art",
    "sota",
    "improves",
    "reduces",
    "increases",
]


def _text(item: Dict[str, Any]) -> str:
    return (
        item.get("cleaned_text")
        or item.get("raw_text")
        or item.get("title")
        or ""
    ).strip()


def _extract_facts(text: str, limit: int = 5) -> List[str]:
    facts: List[str] = []
    if not text:
        return facts
    for sent in _SENTENCE_SPLIT.split(text):
        s = sent.strip()
        if not s:
            continue
        lower = s.lower()
        if any(k in lower for k in _FACT_KEYWORDS):
            facts.append(s)
        if len(facts) >= limit:
            break
    return facts


def _extract_metrics(text: str, limit: int = 8) -> List[str]:
    if not text:
        return []
    metrics: List[str] = []
    patterns = [
        r"\b\d+(?:\.\d+)?\s*%+\b",
        r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds)\b",
        r"\b\d+(?:\.\d+)?\s*(?:k|m|b)\s*parameters\b",
        r"\b\d+(?:\.\d+)?\s*(?:tokens|tok/s|tokens/s)\b",
    ]
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            metrics.append(m)
            if len(metrics) >= limit:
                return metrics
    return metrics


def _parse_requirements(text: str) -> List[str]:
    deps: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        deps.append(re.split(r"[<>=~]", line)[0].strip())
    return deps


def _parse_package_json(text: str) -> List[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    deps = []
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        block = data.get(key) or {}
        deps.extend(list(block.keys()))
    return deps


def _parse_pyproject(text: str) -> List[str]:
    if not tomllib:
        return []
    try:
        data = tomllib.loads(text)
    except Exception:
        return []
    deps = []
    project = data.get("project") or {}
    for dep in project.get("dependencies", []) or []:
        deps.append(re.split(r"[<>=~]", dep)[0].strip())
    return deps


def _parse_cargo(text: str) -> List[str]:
    deps: List[str] = []
    in_deps = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_deps = s == "[dependencies]"
            continue
        if not in_deps or not s or s.startswith("#"):
            continue
        name = s.split("=", 1)[0].strip()
        if name:
            deps.append(name)
    return deps


def _parse_manifest(item: Dict[str, Any]) -> List[str]:
    text = item.get("manifest_text") or ""
    mtype = (item.get("manifest_type") or "").lower()
    if not text:
        return []
    if mtype == "requirements.txt":
        return _parse_requirements(text)
    if mtype == "package.json":
        return _parse_package_json(text)
    if mtype == "pyproject.toml":
        return _parse_pyproject(text)
    if mtype == "cargo.toml":
        return _parse_cargo(text)
    return []


def run(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in items:
        text = _text(item)
        item["facts"] = _extract_facts(text)
        item["metrics"] = _extract_metrics(text)
        deps = _parse_manifest(item)
        if deps:
            item["dependencies"] = deps
    return items
