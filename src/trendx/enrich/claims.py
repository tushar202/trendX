from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel, Field

from trendx.utils.logging import get_logger
from trendx.utils.json_extract import extract_json
from trendx.utils.llm_debug import log_llm_event

logger = get_logger(__name__)


class Anchor(BaseModel):
    anchor_text: str
    anchor_type: str = "snippet"
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    confidence: Optional[float] = None


class Claim(BaseModel):
    claim_text: str
    evidence_score: Optional[float] = None
    novelty_score: Optional[float] = None
    anchors: List[Anchor] = Field(default_factory=list)


class ClaimExtraction(BaseModel):
    claims: List[Claim] = Field(default_factory=list)


def _model_name(provider: str, model: str) -> str:
    if "/" in model:
        return model
    return f"{provider}/{model}" if provider else model


def _clip_text(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _normalize_claims(data: Any) -> ClaimExtraction:
    if isinstance(data, list):
        data = {"claims": data}
    if not isinstance(data, dict):
        return ClaimExtraction(claims=[])

    claims = data.get("claims") or []
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_text = claim.get("claim_text") or claim.get("claim") or claim.get("text")
        if not claim_text:
            continue

        normalized_claim: Dict[str, Any] = {"claim_text": claim_text}
        if isinstance(claim.get("evidence_score"), (int, float)):
            normalized_claim["evidence_score"] = float(claim["evidence_score"])
        if isinstance(claim.get("novelty_score"), (int, float)):
            normalized_claim["novelty_score"] = float(claim["novelty_score"])

        anchors = claim.get("anchors") or claim.get("evidence") or []
        fixed_anchors = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            anchor_text = anchor.get("anchor_text") or anchor.get("text")
            if not anchor_text:
                continue
            fixed_anchor: Dict[str, Any] = {"anchor_text": anchor_text}
            anchor_type = anchor.get("anchor_type") or anchor.get("type")
            if anchor_type:
                if anchor_type not in {
                    "snippet",
                    "quote",
                    "figure",
                    "table",
                    "code",
                    "other",
                }:
                    anchor_type = "other"
                fixed_anchor["anchor_type"] = anchor_type
            if isinstance(anchor.get("start_offset"), int):
                fixed_anchor["start_offset"] = anchor["start_offset"]
            elif isinstance(anchor.get("start"), int):
                fixed_anchor["start_offset"] = anchor["start"]
            if isinstance(anchor.get("end_offset"), int):
                fixed_anchor["end_offset"] = anchor["end_offset"]
            elif isinstance(anchor.get("end"), int):
                fixed_anchor["end_offset"] = anchor["end"]
            if isinstance(anchor.get("confidence"), (int, float)):
                fixed_anchor["confidence"] = float(anchor["confidence"])
            fixed_anchors.append(fixed_anchor)
        if fixed_anchors:
            normalized_claim["anchors"] = fixed_anchors
        normalized.append(normalized_claim)

    try:
        return ClaimExtraction.model_validate({"claims": normalized})
    except Exception:
        return ClaimExtraction(claims=[])


async def _extract_claims_llm(
    title: str,
    body: str,
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
) -> ClaimExtraction:
    system = (
        "You extract 1-3 concrete claims and evidence anchors. "
        "Return JSON only with a top-level 'claims' list."
    )
    user = {
        "title": title,
        "body": body,
        "instructions": [
            "Each claim should be a single sentence.",
            "Provide 1-2 anchors per claim using direct snippet text.",
            "Set anchor_type to snippet/quote/figure/table/code.",
            "Optionally include evidence_score (0-1) and novelty_score (0-1).",
        ],
    }

    api_key = os.getenv(api_key_env) if api_key_env else None
    if not api_key and provider != "ollama":
        logger.warning("claim extraction skipped: missing API key")
        return ClaimExtraction(claims=[])

    request_payload = {
        "system": system,
        "user": user,
    }
    resp = await litellm.acompletion(
        model=_model_name(provider, model),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
        api_key=api_key,
        api_base=base_url if base_url else None,
    )
    content = resp["choices"][0]["message"]["content"]
    log_llm_event(
        stage="claims",
        model=_model_name(provider, model),
        request=request_payload,
        response={"content": content},
    )
    data = extract_json(content)
    if data is None:
        logger.warning("claim extraction JSON parse failed")
        return ClaimExtraction(claims=[])
    parsed = _normalize_claims(data)
    if not parsed.claims:
        logger.warning("claim extraction JSON validation failed")
    return parsed


async def enrich_items_with_claims(
    items: List[Dict[str, Any]],
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
) -> List[Dict[str, Any]]:
    for item in items:
        title = item.get("title") or ""
        body = _clip_text(item.get("cleaned_text") or item.get("raw_text") or "")
        if not (title or body):
            continue
        try:
            extraction = await _extract_claims_llm(
                title, body, provider, model, api_key_env, base_url
            )
            item["claims"] = [c.model_dump(exclude_none=True) for c in extraction.claims]
        except Exception as exc:
            logger.warning("claim extraction failed: %s", exc)
    return items
