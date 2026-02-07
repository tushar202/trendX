from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from trendx.config import AppConfig
from trendx.formatter.render import render_report
from trendx.pipeline import (
    PipelineContext,
    cluster,
    critique,
    deduplicate,
    discover,
    enrich,
    filter_triage,
    ingest,
    normalize,
    synthesis,
    trend_detect,
    write_report,
)
from trendx.agents.synthesis import refine_single_trend


async def run_ingest_flow(
    config: AppConfig, week: str, run_dir: str
) -> List[Dict[str, Any]]:
    ctx = PipelineContext(config=config, week=week)
    items = await discover(ctx)
    items = await ingest(ctx, items)
    items = await normalize(ctx, items)
    items = await filter_triage(ctx, items)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, "items.json").write_text(json.dumps(items), encoding="utf-8")
    return items


async def run_clustering_flow(
    config: AppConfig, week: str, run_dir: str, items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    ctx = PipelineContext(config=config, week=week)
    items = await enrich(ctx, items)
    items = await deduplicate(ctx, items)
    clusters = await cluster(ctx, items)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    return clusters


async def run_synthesis_flow(
    config: AppConfig, week: str, clusters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    ctx = PipelineContext(config=config, week=week)
    trends = await trend_detect(ctx, clusters)
    trends = await critique(ctx, trends)
    trends = await synthesis(ctx, trends)
    return trends


async def revise_trend_draft(
    config: AppConfig,
    trend: Dict[str, Any],
    cluster_items: List[Dict[str, Any]],
    feedback: str,
) -> Dict[str, Any]:
    return await refine_single_trend(
        trend=trend,
        cluster_items=cluster_items,
        agentic_cfg=config.agentic_synthesis,
        provider=config.llm.provider,
        model=config.llm.model,
        api_key_env=config.llm.api_key_env,
        base_url=config.llm.base_url,
        embedding_model=config.embeddings.model,
        external_feedback=feedback,
    )


async def publish_report(
    week: str, run_dir: str, trends: List[Dict[str, Any]]
) -> tuple[Path, Path]:
    report = render_report(week, trends)
    ctx = PipelineContext(config=None, week=week)
    return await write_report(ctx, report, output_dir=run_dir)
