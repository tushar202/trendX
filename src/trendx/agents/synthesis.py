from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import litellm

from trendx.agents.prompts import (
    CRITIC_PROMPT,
    REFINER_DECISION_PROMPT,
    REFINER_WITH_EVIDENCE_PROMPT,
)
from trendx.config import AgenticSynthesisConfig
from trendx.utils.logging import get_logger
from trendx.utils.json_extract import extract_json
from trendx.utils.llm_debug import log_llm_event
from trendx.tools.retrieval import search_cluster_evidence, tavily_search

logger = get_logger(__name__)


def _model_name(provider: str, model: str) -> str:
    if "/" in model:
        return model
    return f"{provider}/{model}" if provider else model


def _evidence_from_trend(trend: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize evidence metrics for a trend (source counts, trust score).
    This blob is passed to the LLM to ground the synthesis.
    """
    counts: Dict[str, int] = {}
    trust_scores: List[float] = []
    conflicts = 0
    for item in trend.get("items", []) or []:
        src = item.get("source") or "unknown"
        counts[src] = counts.get(src, 0) + 1
        if item.get("trust_score") is not None:
            trust_scores.append(float(item["trust_score"]))
        if item.get("conflict"):
            conflicts += 1
    trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
    return {
        "source_counts": counts,
        "trust_score": round(trust, 3),
        "conflicts": conflicts,
    }


def _fallback_synthesis(trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate basic placeholders if LLM synthesis fails or is disabled.
    Useful for testing or when API limits are hit.
    """
    out: List[Dict[str, Any]] = []
    for t in trends:
        label = t.get("cluster_label") or t.get("label") or "Untitled"
        if label.lower() in {"miscellaneous", "general"}:
            titles = [i.get("title") for i in (t.get("items") or []) if i.get("title")]
            if titles:
                label = "Mixed: " + "; ".join(titles[:2])
        novelty = t.get("novelty_score")
        trend_type = t.get("trend_type", "continuing")
        summary = (
            f"{label} appears {trend_type}. "
            f"Novelty score {novelty:.2f}." if isinstance(novelty, (int, float)) else
            f"{label} appears {trend_type}."
        )
        so_what = "Track this area; evaluate applicability to your stack."
        out.append({**t, "title": label, "summary": summary, "so_what": so_what})
    return out


def _extract_claim_snippets(items: Sequence[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Extract concrete claim text and anchors from items to feed the LLM.
    Limits to 'limit' items to save context window.
    """
    snippets: List[Dict[str, Any]] = []
    for item in items:
        for claim in item.get("claims") or []:
            claim_text = claim.get("claim_text")
            if not claim_text:
                continue
            anchors = []
            for anchor in claim.get("anchors") or []:
                text = anchor.get("anchor_text")
                if text:
                    anchors.append(text)
            snippets.append(
                {
                    "claim": claim_text,
                    "evidence": anchors[:2],
                    "source": item.get("source"),
                }
            )
            if len(snippets) >= limit:
                return snippets
    return snippets


def _normalize_synthesis_output(data: Any) -> List[Dict[str, Any]] | None:
    """Ensure LLM output matches expected schema (List of dicts with title, summary, so_what)."""
    if isinstance(data, dict) and "trends" in data:
        data = data["trends"]
    if not isinstance(data, list):
        return None
    cleaned: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title")
        summary = entry.get("summary")
        so_what = entry.get("so_what") or entry.get("soWhat")
        if not all(isinstance(v, str) and v.strip() for v in [title, summary, so_what]):
            continue
        cleaned.append({"title": title.strip(), "summary": summary.strip(), "so_what": so_what.strip()})
    return cleaned if cleaned else None


def _format_evidence_blob(trend: Dict[str, Any]) -> str:
    evidence = trend.get("evidence") or {}
    claims = _extract_claim_snippets(trend.get("items") or [], limit=3)
    return json.dumps({"evidence": evidence, "claims": claims}, ensure_ascii=False)


async def _call_llm(
    stage: str,
    system: str,
    user: str,
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
    temperature: float = 0.2,
) -> str:
    """Wrapper for LLM calls with logging."""
    api_key = os.getenv(api_key_env) if api_key_env else None
    if not api_key and provider != "ollama":
        raise RuntimeError("missing API key")

    resp = await litellm.acompletion(
        model=_model_name(provider, model),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        api_key=api_key,
        api_base=base_url if base_url else None,
    )
    content = resp["choices"][0]["message"]["content"]
    log_llm_event(
        stage=stage,
        model=_model_name(provider, model),
        request={"system": system, "user": user},
        response={"content": content},
    )
    return content


def _clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except Exception:
        return 0
    return max(0, min(10, score))


def _normalize_rewrite(rewrite: Dict[str, Any], current: Dict[str, str]) -> Dict[str, str]:
    title = rewrite.get("title") or current.get("title") or "Untitled"
    summary = rewrite.get("summary") or current.get("summary") or ""
    so_what = rewrite.get("so_what") or current.get("so_what") or ""
    return {
        "title": str(title).strip(),
        "summary": str(summary).strip(),
        "so_what": str(so_what).strip(),
    }


async def _refine_single_trend(
    trend: Dict[str, Any],
    agentic_cfg: AgenticSynthesisConfig,
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
    embedding_model: str,
    external_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Agentic Loop: Evaluates and refines a single trend draft.
    """
    # Initialize current draft from input
    current = {
        "title": trend.get("title") or trend.get("cluster_label") or "Untitled",
        "summary": trend.get("summary") or "",
        "so_what": trend.get("so_what") or "",
    }
    retries = max(0, int(agentic_cfg.max_retries))
    threshold = _clamp_score(agentic_cfg.quality_threshold)
    critic_model = agentic_cfg.critic_model or model

    # Format the evidence context once
    evidence_blob = _format_evidence_blob(trend)

    for attempt in range(retries):
        # ---------------------------------------------------------------------
        # PHASE 1: CRITIQUE & SCORE
        # ---------------------------------------------------------------------
        if external_feedback and attempt == 0:
            # External feedback (from Orchestrator/Manager) overrides the LLM critic for the first attempt.
            # This ensures the agent acts on the specific instruction first, whether it's
            # from a human override or an automated policy enforcement.
            score = 0
            feedback = f"MANAGEMENT OVERRIDE: {external_feedback}"
        else:
            try:
                critique_prompt = CRITIC_PROMPT.format(
                    title=current["title"],
                    summary=current["summary"],
                    so_what=current["so_what"],
                    evidence=evidence_blob,
                )
                critique_raw = await _call_llm(
                    stage="synthesis_critic",
                    system="Return JSON only.",
                    user=critique_prompt,
                    provider=provider,
                    model=critic_model,
                    api_key_env=api_key_env,
                    base_url=base_url,
                    temperature=0.0,
                )
                critique = extract_json(critique_raw) or {}
                score = _clamp_score(critique.get("score"))
                feedback = critique.get("feedback") or "Improve clarity and actionability."
            except Exception as exc:
                logger.warning("synthesis critique failed: %s", exc)
                return current

        # Check if quality threshold is met (Exit Condition)
        if score >= threshold:
            logger.info("Trend met quality threshold (%s/%s). Returning.", score, threshold)
            return current

        logger.info("refining trend (score=%s/%s). Feedback: %s", score, threshold, feedback)
        
        # ---------------------------------------------------------------------
        # PHASE 2: REFINEMENT STRATEGY (DECISION)
        # ---------------------------------------------------------------------
        try:
            # Ask the Refiner: "Given this feedback, should we SEARCH or REWRITE?"
            decision_prompt = REFINER_DECISION_PROMPT.format(
                title=current["title"],
                summary=current["summary"],
                so_what=current["so_what"],
                feedback=feedback,
            )
            decision_raw = await _call_llm(
                stage="synthesis_refiner_decision",
                system="Return JSON only.",
                user=decision_prompt,
                provider=provider,
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=0.2,
            )
            decision = extract_json(decision_raw)
            if not isinstance(decision, dict):
                return current

            action = decision.get("action") or "rewrite"
            
            # -----------------------------------------------------------------
            # PHASE 3: EXECUTE ACTION
            # -----------------------------------------------------------------
            
            # Action A: SEARCH (Local Context Only)
            if action == "search" and agentic_cfg.retrieval_enabled:
                query = decision.get("query") or ""
                if not query.strip():
                    # Fallback to rewrite if query is empty
                    rewrite = decision.get("rewrite_content")
                    if isinstance(rewrite, dict):
                        current = _normalize_rewrite(rewrite, current)
                    return current

                # Perform Local Vector Search
                local_evidence = search_cluster_evidence(
                    trend.get("items") or [],
                    query,
                    model_name=embedding_model,
                )

                # Perform External Web Search (New!)
                tavily_key = agentic_cfg.get("tavily_api_key") or config.llm.tavily_api_key
                web_evidence = await tavily_search(query, api_key=tavily_key)

                # Combine Evidence
                combined_evidence = (
                    f"LOCAL EVIDENCE (Cluster Items):\n{local_evidence}\n\n"
                    f"EXTERNAL WEB EVIDENCE (Fresh Search):\n{web_evidence}"
                )
                
                # Rewrite using new evidence
                final_prompt = REFINER_WITH_EVIDENCE_PROMPT.format(
                    evidence=combined_evidence,
                    feedback=feedback,
                    title=current["title"],
                )
                final_raw = await _call_llm(
                    stage="synthesis_refiner_evidence",
                    system="Return JSON only.",
                    user=final_prompt,
                    provider=provider,
                    model=model,
                    api_key_env=api_key_env,
                    base_url=base_url,
                    temperature=0.2,
                )
                final_content = extract_json(final_raw)
                if isinstance(final_content, dict):
                    current = _normalize_rewrite(final_content, current)
                return current

            # Action B: REWRITE (Direct)
            if action == "rewrite":
                rewrite = decision.get("rewrite_content")
                if isinstance(rewrite, dict):
                    current = _normalize_rewrite(rewrite, current)
                return current
                
        except Exception as exc:
            logger.warning("synthesis refine failed: %s", exc)
            return current
    return current


async def refine_single_trend(
    trend: Dict[str, Any],
    cluster_items: List[Dict[str, Any]],
    agentic_cfg: AgenticSynthesisConfig,
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
    embedding_model: str,
    external_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public wrapper to refine a trend using its items.
    
    LIMITATION: The 'search' action here calls `search_cluster_evidence`, which
    ONLY searches within the items already in the cluster. It does NOT search
    the web or external sources.
    IMPACT: If the cluster itself is shallow (few items, weak content), the
    synthesis will also be shallow because the agent cannot find *new* facts.
    
    ROADMAP: Implement a true `Deep Synthesis Agent`:
    1. Allow 'search' to call a web search tool (e.g., Tavily, Serper).
    2. Implement 'Gap Analysis': Ask LLM "What is missing?" before searching.
    3. Use 'comparative analysis' prompts to contrast items.
    """
    # Step 1: Clone the trend to avoid mutating the original until success
    trend_payload = {**trend}
    
    # Step 2: Attach the raw items (needed for 'search' actions)
    trend_payload["items"] = cluster_items
    
    # Step 3: Pre-calculate evidence stats (source counts, trust scores)
    # This gives the agent a "meta-view" of the data quality.
    trend_payload["evidence"] = _evidence_from_trend({"items": cluster_items})
    
    # Step 4: Enter the Agentic Loop
    return await _refine_single_trend(
        trend_payload,
        agentic_cfg,
        provider,
        model,
        api_key_env,
        base_url,
        embedding_model,
        external_feedback=external_feedback,
    )


async def _request_synthesis_json(
    system: str,
    user: Dict[str, Any],
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
    max_retries: int = 2,
) -> List[Dict[str, Any]] | None:
    """Make the initial LLM call to generate drafts for all trends."""
    api_key = os.getenv(api_key_env) if api_key_env else None
    if not api_key and provider != "ollama":
        logger.warning("synthesis skipped: missing API key")
        return None

    request_payload = {"system": system, "user": user}
    for attempt in range(1, max_retries + 1):
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
            stage="synthesis",
            model=_model_name(provider, model),
            request=request_payload,
            response={"content": content},
            extra={"attempt": attempt},
        )
        data = extract_json(content)
        normalized = _normalize_synthesis_output(data)
        if normalized is not None:
            return normalized

        # Tighten prompt for retry if JSON parsing failed
        system = (
            "Return ONLY valid JSON. No markdown, no code fences. "
            "Return a JSON array of objects with keys: title, summary, so_what."
        )
    return None


async def run(
    trends: List[Dict[str, Any]],
    provider: str,
    model: str,
    api_key_env: Optional[str],
    base_url: Optional[str],
    agentic_cfg: Optional[AgenticSynthesisConfig] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Main Synthesis Pipeline:
    1. Pre-process trends (extract evidence, claims).
    2. Batch LLM call to generate initial drafts.
    3. (Optional) Agentic Refinement loop for each trend.
    """
    if not trends:
        return trends

    # Pre-calculate evidence stats
    for t in trends:
        t["evidence"] = _evidence_from_trend(t)

    fallback = _fallback_synthesis([t.copy() for t in trends])

    # Prepare payload for Batch LLM generation
    payload = []
    for t in trends:
        claims = _extract_claim_snippets(t.get("items") or [])
        sample_items = []
        for item in (t.get("items") or [])[:5]:
            sample_items.append(
                {
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                }
            )
        payload.append(
            {
                "cluster_label": t.get("cluster_label") or t.get("label"),
                "trend_type": t.get("trend_type"),
                "novelty_score": t.get("novelty_score"),
                "confidence": t.get("confidence"),
                "evidence": t.get("evidence"),
                "claims": claims,
                "items": sample_items,
            }
        )

    system = (
        "You are a synthesis agent. Return ONLY valid JSON (no markdown). "
        "Output must be a JSON array of objects with fields: title, summary, so_what. "
        "Keep summary to 2-3 sentences and so_what to 1 sentence. "
        "Use provided claims/evidence; do not hallucinate facts."
    )
    user = {"trends": payload}

    try:
        # Phase 1: Batch Generation
        data = await _request_synthesis_json(
            system,
            user,
            provider,
            model,
            api_key_env,
            base_url,
            max_retries=2,
        )
        if data is None:
            raise ValueError("synthesis output not parseable")

        # Merge results into trends
        for idx, t in enumerate(trends):
            base = fallback[idx] if idx < len(fallback) else {}
            if idx < len(data):
                t["title"] = data[idx].get("title") or base.get("title")
                t["summary"] = data[idx].get("summary") or base.get("summary")
                t["so_what"] = data[idx].get("so_what") or base.get("so_what")
            else:
                t["title"] = base.get("title")
                t["summary"] = base.get("summary")
                t["so_what"] = base.get("so_what")
                
        # Phase 2: Refinement (Critique -> Improve)
        if agentic_cfg and agentic_cfg.enabled:
            refined_trends = []
            for t in trends:
                refined = await _refine_single_trend(
                    t,
                    agentic_cfg,
                    provider,
                    model,
                    api_key_env,
                    base_url,
                    embedding_model,
                )
                t["title"] = refined["title"]
                t["summary"] = refined["summary"]
                t["so_what"] = refined["so_what"]
                refined_trends.append(t)
            return refined_trends
        return trends
    except Exception as exc:
        logger.warning("synthesis failed, fallback: %s", exc)
        return fallback
