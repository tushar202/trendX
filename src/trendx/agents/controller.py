from __future__ import annotations

import json
import os
from typing import Any, Dict

import litellm

from trendx.config import AppConfig
from trendx.utils.json_extract import extract_json
from trendx.utils.logging import get_logger

logger = get_logger(__name__)

CONTROLLER_SYSTEM_PROMPT = """
You are the Orchestrator. Your goal is to publish a Weekly Trend Report.

CURRENT STATE:
Step: {step}
Drafts Pending Review: {pending_count}
Drafts:
{drafts_json}

RULES:
1. If step is 'start', run ingest.
2. If step is 'ingest', run clustering.
3. If step is 'cluster', run synthesis.
4. If step is 'audit', review the drafts:
   - REJECT if: uses "I/We/Our", vague claims ("significant improvement" without numbers), or sounds like a paper author.
   - APPROVE if: neutral tone, specific tool names, actionable "So What".
   - You can reject MULTIPLE trends in one turn.
5. If all approved, publish.

RETURN JSON (Strict):
{{
  "action": "run_ingest" | "run_clustering" | "run_synthesis" | "review_drafts" | "publish",
  "reviews": [
    {{"draft_id": "trend_1", "status": "approved"}},
    {{"draft_id": "trend_2", "status": "rejected", "feedback": "Used 'Our analysis'. Rewrite in third person."}}
  ]
}}
"""


def auto_audit(text: str) -> str:
    errors = []
    lower = text.lower()
    if any(x in lower for x in ["our analysis", "we propose", "in this paper", "my opinion"]):
        errors.append("Persona Error: Used first-person (We/Our). Must be third-person.")
    if "significant improvement" in lower and "%" not in lower and "x" not in lower:
        errors.append("Vagueness Error: Claimed improvement without numbers.")
    return "; ".join(errors) if errors else "PASS"


def _model_name(provider: str, model: str) -> str:
    if "/" in model:
        return model
    return f"{provider}/{model}" if provider else model


async def _call_llm(prompt: str, config: AppConfig) -> str:
    api_key = os.getenv(config.llm.api_key_env) if config.llm.api_key_env else None
    if not api_key and config.llm.provider != "ollama":
        raise RuntimeError("missing API key")

    resp = await litellm.acompletion(
        model=_model_name(config.llm.provider, config.llm.model),
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        api_key=api_key,
        api_base=config.llm.base_url if config.llm.base_url else None,
    )
    return resp["choices"][0]["message"]["content"]


async def get_next_action(state, config: AppConfig) -> Dict[str, Any]:
    if state.step == "audit":
        for draft in state.drafts:
            if draft.status == "draft":
                error = auto_audit(draft.summary + " " + draft.so_what)
                if error != "PASS":
                    draft.status = "rejected"
                    draft.feedback_history.append(f"[Auto-Audit] {error}")

    drafts_payload = [
        {
            "id": d.id,
            "cluster_label": d.cluster_label,
            "title": d.title,
            "summary": d.summary,
            "so_what": d.so_what,
            "status": d.status,
        }
        for d in state.drafts
    ]

    prompt = CONTROLLER_SYSTEM_PROMPT.format(
        step=state.step,
        pending_count=len([d for d in state.drafts if d.status == "draft"]),
        drafts_json=json.dumps(drafts_payload, ensure_ascii=False),
    )

    try:
        raw = await _call_llm(prompt, config)
    except Exception as exc:
        logger.warning("controller call failed: %s", exc)
        return {"action": "review_drafts", "reviews": []}

    decision = extract_json(raw)
    if not decision:
        return {"action": "review_drafts", "reviews": []}
    return decision
