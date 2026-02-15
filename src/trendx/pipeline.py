from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from trendx.config import AppConfig, load_config
from trendx.formatter.render import render_report
from trendx.ingest.arxiv import fetch_arxiv_items
from trendx.ingest.github import fetch_repo_metadata, repo_full_name_from_item
from trendx.dedupe.basic import dedupe_items
from trendx.enrich.claims import enrich_items_with_claims
from trendx.enrich.trust import apply_trust_scores
from trendx.filter.triage import filter_items
from trendx.ingest.github_source import fetch_github_repos
from trendx.normalize.basic import normalize_items
from trendx.agents.critic import run as critic_run
from trendx.agents.evidence_anchor import run as evidence_anchor_run
from trendx.agents.research import run as research_run
from trendx.agents.synthesis import run as synthesis_run
from trendx.agents.trend_detector import run as trend_detector_run
from trendx.trend.taxonomy import TaxonomyConfig, cluster_with_taxonomy
from trendx.storage.db import get_engine, get_session_factory, init_db
from trendx.storage.persist import (
    save_claims,
    save_clusters,
    save_evidence_anchors,
    save_items,
    save_repo_metadata,
    save_report,
    save_trends,
    save_validation_logs,
)
from trendx.utils.logging import get_logger
from trendx.utils.litellm_runtime import configure_litellm
from trendx.utils.llm_debug import enable_debug, log_run_metadata
from trendx.utils.time import week_id
from trendx.validate.validator import validate_payload

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    config: AppConfig
    week: str


async def discover(ctx: PipelineContext) -> List[Dict[str, Any]]:
    logger.info("discover: building candidate list")
    items: List[Dict[str, Any]] = []
    if ctx.config.sources.papers.enabled:
        query = ctx.config.sources.papers.arxiv.get("query", "agentic ai")
        max_results = ctx.config.run.max_items_per_source
        items.extend(await fetch_arxiv_items(query, max_results=max_results))
    if ctx.config.sources.github.enabled and ctx.config.sources.github.trending:
        query = ctx.config.sources.github.search_query or "agentic ai"
        max_results = ctx.config.run.max_items_per_source
        items.extend(await fetch_github_repos(query, max_results=max_results))
    return items


async def ingest(ctx: PipelineContext, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("ingest: fetching content and repo metadata")
    # Enrich GitHub items with repo metadata where possible
    for item in items:
        repo_full_name = repo_full_name_from_item(item)
        if not repo_full_name:
            continue
        try:
            meta = await fetch_repo_metadata(repo_full_name)
            item["repo_metadata"] = meta
        except Exception as exc:
            logger.warning("repo metadata fetch failed: %s", exc)
    return items


async def normalize(ctx: PipelineContext, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("normalize: canonicalizing fields")
    return normalize_items(items)


async def filter_triage(ctx: PipelineContext, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("filter: dropping low-signal items")
    return filter_items(items)


async def enrich(ctx: PipelineContext, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("enrich: embeddings, claims, trust scoring, evidence anchors")
    items = research_run(items)
    items = await enrich_items_with_claims(
        items,
        provider=ctx.config.llm.provider,
        model=ctx.config.llm.model,
        api_key_env=ctx.config.llm.api_key_env,
        base_url=ctx.config.llm.base_url,
    )
    items = evidence_anchor_run(items)
    apply_trust_scores(items, ctx.config.trust.source_weights)
    return items


async def deduplicate(ctx: PipelineContext, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("dedupe: canonical URL + hash + similarity")
    return dedupe_items(items, ctx.config.embeddings.model)


async def cluster(ctx: PipelineContext, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("cluster: LLM taxonomy + vector assignment")
    cfg = TaxonomyConfig(
        llm_provider=ctx.config.llm.provider,
        llm_model=ctx.config.llm.model,
        api_key_env=ctx.config.llm.api_key_env,
        base_url=ctx.config.llm.base_url,
        embedding_model=ctx.config.embeddings.model,
    )
    return await cluster_with_taxonomy(ctx.week, items, cfg)


async def trend_detect(ctx: PipelineContext, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("trend: compare against history")
    trends = await trend_detector_run(ctx.config.storage.url, ctx.week, clusters)
    # Apply corroboration factor after clustering
    for t in trends:
        for item in t.get("items", []) or []:
            if "corroboration_factor" in item:
                continue
        apply_trust_scores(t.get("items", []) or [], ctx.config.trust.source_weights)
    return trends


async def critique(ctx: PipelineContext, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("critique: hype, contradictions, confidence penalties")
    return critic_run(trends)


async def synthesis(ctx: PipelineContext, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("synthesis: build narratives and action lines")
    return await synthesis_run(
        trends,
        provider=ctx.config.llm.provider,
        model=ctx.config.llm.model,
        api_key_env=ctx.config.llm.api_key_env,
        base_url=ctx.config.llm.base_url,
        agentic_cfg=ctx.config.agentic_synthesis,
        embedding_model=ctx.config.embeddings.model,
    )


async def formatter(ctx: PipelineContext, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("formatter: render Markdown and JSON")
    return render_report(ctx.week, trends)


def _week_ids_for_backfill(weeks: int) -> List[str]:
    today = date.today()
    ids = []
    for i in range(weeks):
        d = today - timedelta(weeks=i)
        ids.append(week_id(d))
    return list(reversed(ids))


async def run_pipeline(config_path: str, week: str | None = None) -> None:
    config = load_config(config_path)
    configure_litellm(config.llm)
    engine = get_engine(config.storage.url)
    await init_db(engine)

    ctx = PipelineContext(config=config, week=week or week_id())

    if ctx.config.llm.debug:
        debug_path = ctx.config.llm.debug_path or f"reports/{ctx.week}/llm_debug.jsonl"
        enable_debug(debug_path)

    validation_records: List[Dict[str, Any]] = []

    def _validate(stage: str, payload: Dict[str, Any]) -> None:
        ok, errors = validate_payload(stage, payload)
        validation_records.append(
            {
                "stage": stage,
                "status": "ok" if ok else "error",
                "errors": errors,
            }
        )

    items = await discover(ctx)
    _validate("discover", {"items": items})

    items = await ingest(ctx, items)
    _validate("ingest", {"items": items})

    items = await normalize(ctx, items)
    _validate("normalize", {"items": items})

    items = await filter_triage(ctx, items)
    _validate("filter", {"items": items})

    items = await enrich(ctx, items)
    _validate("enrich", {"items": items})

    items = await deduplicate(ctx, items)
    _validate("dedupe", {"items": items})

    clusters = await cluster(ctx, items)
    _validate("cluster", {"clusters": clusters})

    trends = await trend_detect(ctx, clusters)
    _validate("trend", {"trends": trends})

    trends = await critique(ctx, trends)
    _validate("critique", {"trends": trends})

    trends = await synthesis(ctx, trends)
    _validate("synthesis", {"trends": trends})

    report = await formatter(ctx, trends)
    _validate("report", report)

    md_path, json_path = await write_report(ctx, report)

    # Persist entities
    session_factory = get_session_factory(engine)
    async with session_factory() as session:
        item_map = await save_items(session, items)
        await save_repo_metadata(session, items, item_map)
        claim_map = await save_claims(session, items, item_map)
        await save_evidence_anchors(session, items, item_map, claim_map)
        cluster_map = await save_clusters(session, ctx.week, clusters, item_map)
        await save_trends(session, ctx.week, trends, cluster_map)
        await save_report(session, ctx.week, str(md_path), str(json_path))
        await save_validation_logs(session, validation_records)
        await session.commit()

    if ctx.config.llm.debug:
        log_run_metadata(
            {
                "week": ctx.week,
                "markdown": str(md_path),
                "json": str(json_path),
            }
        )


async def write_report(
    ctx: PipelineContext, report: Dict[str, Any], output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    if output_dir is not None:
        base = Path(output_dir)
    else:
        base = Path("reports") / ctx.week
        if (base / "brief.md").exists() or (base / "brief.json").exists():
            run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
            base = base / run_id
    base.mkdir(parents=True, exist_ok=True)

    md_path = base / "brief.md"
    json_path = base / "brief.json"

    md_path.write_text(report["markdown"], encoding="utf-8")
    json_path.write_text(report["json"], encoding="utf-8")

    logger.info("report written: %s", base)
    return md_path, json_path


async def run_backfill(config_path: str, weeks: int) -> None:
    for wid in _week_ids_for_backfill(weeks):
        await run_pipeline(config_path, week=wid)
        await asyncio.sleep(0)
