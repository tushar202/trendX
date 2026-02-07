from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from trendx.storage.models import (
    Claim,
    Cluster,
    ClusterItem,
    EvidenceAnchor,
    Item,
    RepoMetadata,
    Report,
    Trend,
    ValidationLog,
)


def _item_key(item: Dict[str, Any]) -> str:
    return (item.get("url") or item.get("canonical_url") or "").strip()


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


async def save_items(session: AsyncSession, items: List[Dict[str, Any]]) -> Dict[str, Item]:
    mapping: Dict[str, Item] = {}
    for item in items:
        key = _item_key(item)
        if not key:
            continue

        if isinstance(item.get("id"), int):
            existing = await session.get(Item, item["id"])
            if existing:
                mapping[key] = existing
                continue

        stmt = select(Item).where(Item.url == key)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            mapping[key] = existing
            continue

        obj = Item(
            url=key,
            canonical_url=item.get("canonical_url"),
            title=item.get("title"),
            authors=item.get("authors"),
            published_at=_parse_dt(item.get("published_at")),
            raw_text=item.get("raw_text"),
            cleaned_text=item.get("cleaned_text"),
            hash=item.get("hash"),
            embedding=item.get("embedding"),
            trust_score=item.get("trust_score"),
        )
        session.add(obj)
        await session.flush()
        mapping[key] = obj

    return mapping


async def save_repo_metadata(
    session: AsyncSession, items: List[Dict[str, Any]], item_map: Dict[str, Item]
) -> None:
    for item in items:
        key = _item_key(item)
        if not key or key not in item_map:
            continue
        meta = item.get("repo_metadata") or item.get("github_metadata")
        if not isinstance(meta, dict) or not meta:
            continue
        obj = RepoMetadata(
            item_id=item_map[key].id,
            stars=meta.get("stars"),
            forks=meta.get("forks"),
            created_at=_parse_dt(meta.get("created_at")),
            updated_at=_parse_dt(meta.get("updated_at")),
            license=meta.get("license"),
            has_tests=bool(meta.get("has_tests") or meta.get("has_tests_folder")),
            has_dockerfile=bool(meta.get("has_dockerfile")),
            has_docker_compose=bool(meta.get("has_docker_compose")),
        )
        session.add(obj)


async def save_clusters(
    session: AsyncSession,
    week: str,
    clusters: List[Dict[str, Any]],
    item_map: Dict[str, Item],
) -> Dict[str, Cluster]:
    cluster_map: Dict[str, Cluster] = {}
    for cluster in clusters:
        label = cluster.get("label") or "Untitled"
        obj = Cluster(
            week=week,
            label=label,
            centroid_embedding=cluster.get("centroid_embedding"),
            summary=cluster.get("summary"),
        )
        session.add(obj)
        await session.flush()
        cluster_map[label] = obj

        for item in cluster.get("items", []):
            key = _item_key(item)
            if not key:
                continue
            item_obj = item_map.get(key)
            if not item_obj:
                continue
            session.add(ClusterItem(cluster_id=obj.id, item_id=item_obj.id))

    return cluster_map


async def save_trends(
    session: AsyncSession,
    week: str,
    trends: List[Dict[str, Any]],
    cluster_map: Dict[str, Cluster],
) -> None:
    for trend in trends:
        label = trend.get("cluster_label") or trend.get("label")
        cluster_id = trend.get("cluster_id")
        if cluster_id is None and label in cluster_map:
            cluster_id = cluster_map[label].id
        if cluster_id is None:
            continue

        obj = Trend(
            week=week,
            cluster_id=int(cluster_id),
            trend_type=trend.get("trend_type", "unknown"),
            novelty_score=trend.get("novelty_score"),
            confidence=trend.get("confidence"),
        )
        session.add(obj)


async def save_report(
    session: AsyncSession, week: str, md_path: str, json_path: str
) -> None:
    session.add(Report(week=week, md_path=md_path, json_path=json_path))


def report_paths(md_path: str, json_path: str) -> Tuple[str, str]:
    return md_path, json_path


async def save_claims(
    session: AsyncSession, items: List[Dict[str, Any]], item_map: Dict[str, Item]
) -> Dict[Tuple[int, str], Claim]:
    claim_map: Dict[Tuple[int, str], Claim] = {}
    for item in items:
        key = _item_key(item)
        if not key or key not in item_map:
            continue
        item_obj = item_map[key]
        for claim in item.get("claims", []) or []:
            text = (claim.get("claim_text") or "").strip()
            if not text:
                continue
            obj = Claim(
                item_id=item_obj.id,
                claim_text=text,
                evidence_score=claim.get("evidence_score"),
                novelty_score=claim.get("novelty_score"),
            )
            session.add(obj)
            await session.flush()
            claim_map[(item_obj.id, text)] = obj
    return claim_map


async def save_evidence_anchors(
    session: AsyncSession,
    items: List[Dict[str, Any]],
    item_map: Dict[str, Item],
    claim_map: Dict[Tuple[int, str], Claim],
) -> None:
    for item in items:
        key = _item_key(item)
        if not key or key not in item_map:
            continue
        item_obj = item_map[key]

        anchors = item.get("evidence_anchors") or []
        # Allow anchors nested under claims
        for claim in item.get("claims", []) or []:
            for anchor in claim.get("anchors", []) or []:
                anchors.append({**anchor, "claim_text": claim.get("claim_text")})

        for anchor in anchors:
            claim_id = anchor.get("claim_id")
            claim_text = (anchor.get("claim_text") or "").strip()
            if claim_id is None and claim_text:
                claim_obj = claim_map.get((item_obj.id, claim_text))
                if claim_obj:
                    claim_id = claim_obj.id
            if claim_id is None:
                continue

            obj = EvidenceAnchor(
                claim_id=int(claim_id),
                item_id=item_obj.id,
                anchor_text=anchor.get("anchor_text") or "",
                anchor_type=anchor.get("anchor_type") or "snippet",
                start_offset=anchor.get("start_offset"),
                end_offset=anchor.get("end_offset"),
                confidence=anchor.get("confidence"),
            )
            session.add(obj)


async def save_validation_logs(
    session: AsyncSession, records: List[Dict[str, Any]]
) -> None:
    for rec in records:
        session.add(
            ValidationLog(
                stage=rec.get("stage", "unknown"),
                item_id=rec.get("item_id"),
                status=rec.get("status", "ok"),
                errors=rec.get("errors"),
            )
        )
