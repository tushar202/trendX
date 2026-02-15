from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from trendx.agents import controller
from trendx.config import load_config
from trendx.state import ReportState, TrendDraft
from trendx.tools import wrappers
from trendx.utils.litellm_runtime import configure_litellm
from trendx.utils.paths import get_new_run_dir
from trendx.utils.time import week_id

logger = logging.getLogger(__name__)

# Maximum number of revision attempts per draft before auto-approving.
# Prevents infinite loops when the Critic threshold exceeds Refiner capability.
MAX_REVISIONS = 3


def _load_json(path: Path) -> List[Dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


async def run_agent(config_path: str, week: str | None = None) -> None:
    """
    The Orchestrator Agent Loop.
    
    This function manages the high-level state of the pipeline:
    1. Steps through stages: Ingest -> Cluster -> Synthesis -> Audit -> Publish.
    2. In 'Audit' mode, it calls the Controller (LLM or Policy) to review drafts.
    3. If the Controller rejects a draft, it calls `wrappers.revise_trend_draft`.
    
    This loop is fully automated (unless configured to wait for human signal), acting
    as the 'Manager' that oversees the other agents.
    """
    config = load_config(config_path)
    configure_litellm(config.llm)
    week = week or week_id()
    run_dir = get_new_run_dir(week)
    run_dir.mkdir(parents=True, exist_ok=True)

    state = ReportState.load(str(run_dir)) or ReportState(week=week, run_dir=str(run_dir))

    items = _load_json(run_dir / "items.json")
    clusters = _load_json(run_dir / "clusters.json")
    trends: List[Dict[str, Any]] | None = None

    while state.step != "done":
        state.save()

        if state.step == "start":
            action = "run_ingest"
        elif state.step == "ingest":
            action = "run_clustering"
        elif state.step == "cluster":
            action = "run_clustering"
        elif state.step == "synthesis":
            action = "run_synthesis"
        elif state.step == "audit":
            decision = await controller.get_next_action(state, config)
            action = decision.get("action")
        elif state.step == "publish":
            action = "publish"
        else:
            action = "review_drafts"

        if action == "run_ingest":
            items = await wrappers.run_ingest_flow(config, state.week, state.run_dir)
            state.items_count = len(items)
            state.step = "cluster"

        elif action == "run_clustering":
            if items is None:
                items = _load_json(Path(state.run_dir) / "items.json") or []
            clusters = await wrappers.run_clustering_flow(
                config, state.week, state.run_dir, items
            )
            state.cluster_count = len(clusters)
            state.step = "synthesis"

        elif action == "run_synthesis":
            if clusters is None:
                clusters = _load_json(Path(state.run_dir) / "clusters.json") or []
            trends = await wrappers.run_synthesis_flow(config, state.week, clusters)
            state.drafts = [
                TrendDraft(
                    id=f"{state.week}:{t.get('cluster_label')}",
                    cluster_label=t.get("cluster_label", "Unknown"),
                    title=t.get("title", ""),
                    summary=t.get("summary", ""),
                    so_what=t.get("so_what", ""),
                )
                for t in trends
            ]
            state.step = "audit"

        elif action == "review_drafts":
            if not state.drafts:
                state.step = "synthesis"
                continue
            reviews = decision.get("reviews", [])
            revise_feedback: Dict[str, str] = {}
            # include auto-audited rejects
            for d in state.drafts:
                if d.status == "rejected":
                    if d.feedback_history:
                        revise_feedback[d.id] = d.feedback_history[-1]
                    else:
                        revise_feedback[d.id] = "Rewrite in neutral, third-person tone with specifics."
            for rev in reviews:
                target = next((d for d in state.drafts if d.id == rev.get("draft_id")), None)
                if not target:
                    continue
                if rev.get("status") == "rejected":
                    target.status = "draft"
                    feedback = rev.get("feedback", "")
                    target.feedback_history.append(feedback)
                    revise_feedback[target.id] = feedback
                else:
                    target.status = "approved"

            if revise_feedback and clusters is not None:
                for draft_id, feedback in revise_feedback.items():
                    target = next((d for d in state.drafts if d.id == draft_id), None)
                    if not target:
                        continue

                    # --- Retry Budget Guard ---
                    if target.revision_count >= MAX_REVISIONS:
                        logger.warning(
                            "Draft '%s' hit max revisions (%d). Auto-approving with review flag.",
                            target.id, MAX_REVISIONS,
                        )
                        target.status = "approved"
                        target.needs_review = True
                        target.feedback_history.append(
                            f"[AUTO-APPROVED] Exceeded {MAX_REVISIONS} revision attempts."
                        )
                        continue

                    cluster = next(
                        (c for c in clusters if c.get("label") == target.cluster_label), None
                    )
                    if not cluster:
                        continue
                    new_data = await wrappers.revise_trend_draft(
                        config,
                        {
                            "title": target.title,
                            "summary": target.summary,
                            "so_what": target.so_what,
                            "cluster_label": target.cluster_label,
                        },
                        cluster.get("items", []),
                        feedback,
                    )
                    target.title = new_data.get("title", target.title)
                    target.summary = new_data.get("summary", target.summary)
                    target.so_what = new_data.get("so_what", target.so_what)
                    target.revision_count += 1
                    target.status = "draft"

            if all(d.status == "approved" for d in state.drafts):
                state.step = "publish"

        elif action == "publish":
            if not state.drafts:
                state.step = "synthesis"
                continue
            if trends is None:
                # Recreate trend dicts from state drafts
                trends = [d.model_dump() for d in state.drafts]
            await wrappers.publish_report(state.week, state.run_dir, trends)
            state.step = "done"

    state.save()
