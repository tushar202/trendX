from __future__ import annotations

from __future__ import annotations

import re
from typing import Any, Dict, List


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def run(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in items:
        text = (item.get("cleaned_text") or item.get("raw_text") or "").strip()
        sents = _sentences(text)
        if not sents:
            continue

        for claim in item.get("claims", []) or []:
            claim_text = (claim.get("claim_text") or "").strip()
            if not claim_text:
                continue
            anchors = []
            for s in sents:
                if any(word.lower() in s.lower() for word in claim_text.split()[:5]):
                    anchors.append(
                        {
                            "anchor_text": s,
                            "anchor_type": "snippet",
                            "confidence": 0.5,
                        }
                    )
                if len(anchors) >= 2:
                    break
            if anchors:
                claim["anchors"] = anchors
    return items
