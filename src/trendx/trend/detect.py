from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from trendx.storage.db import get_engine, get_session_factory
from trendx.storage.models import Cluster
from trendx.utils.logging import get_logger
from trendx.utils.time import week_id

logger = get_logger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (L2 norm)."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _centroid(items: List[Dict[str, Any]]) -> Optional[List[float]]:
    """
    Calculate the centroid of a cluster by averaging item embeddings.
    
    Current Logic:
    - Simple arithmetic mean of all item vectors.
    - Limitation: Does not weight items by importance (e.g., trust score).
    """
    vecs = [i.get("embedding") for i in items if i.get("embedding")]
    if not vecs:
        return None
    arr = np.array(vecs, dtype=float)
    mean = arr.mean(axis=0)
    return _normalize(mean).tolist()


def _prev_week_id(current_week: str) -> str:
    """Calculate the ID of the previous week (YYYY-WW)."""
    year, week = current_week.split("-")
    d = date.fromisocalendar(int(year), int(week), 1)
    prev = d.fromordinal(d.toordinal() - 7)
    return week_id(prev)


async def _load_prev_clusters(session: AsyncSession, prev_week: str) -> List[Cluster]:
    """Load cluster centroids from the previous week for comparison."""
    stmt = select(Cluster).where(Cluster.week == prev_week)
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _best_match_sim(vec: np.ndarray, prev_vecs: List[np.ndarray]) -> float:
    """
    Find the highest cosine similarity between the current cluster and any cluster from the previous week.
    Returns 0.0 if no previous clusters exist.
    """
    if not prev_vecs:
        return 0.0
    sims = [float(vec @ pv) for pv in prev_vecs]
    return max(sims)


def _trend_type(novelty: float, size: int, prev_size: int | None = None) -> str:
    """
    Classify the trend type based on novelty and size changes.
    
    Heuristics:
    - New: Novelty > 0.7 (High dissimilarity to past topics).
    - Declining: Novelty < 0.3 AND shrinking size (if history tracked).
    - Continuing: Everything else.
    """
    if novelty > 0.7:
        return "new"
    if novelty < 0.3 and prev_size is not None and size < prev_size:
        return "declining"
    return "continuing"


def _confidence(size: int, novelty: float) -> float:
    """
    Calculate a confidence score for the trend.
    
    Factors:
    - Size: Larger clusters = higher confidence (up to 1.0 at size 10).
    - Novelty: Bonus 0.1 for highly novel trends (assuming signal strength).
    """
    base = min(1.0, size / 10.0)
    if novelty > 0.7:
        return min(1.0, base + 0.1)
    return base


async def detect_trends(
    db_url: str, week: str, clusters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect trends by comparing current clusters against the previous week's clusters.

    Flow:
    1. Load previous week's clusters from DB.
    2. Compute centroids for current clusters.
    3. For each current cluster:
       - Match against previous week's centroids (Cosine Similarity).
       - Calculate 'Novelty' (1 - Similarity).
       - Classify as 'New', 'Continuing', or 'Declining'.
       - Calculate Confidence Score.
       - Apply Corroboration Factor (source diversity).
    """
    if not clusters:
        return []

    prev_week = _prev_week_id(week)
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)

    # 1. Load History
    async with session_factory() as session:
        prev_clusters = await _load_prev_clusters(session, prev_week)

    prev_vecs: List[np.ndarray] = []
    prev_sizes: Dict[int, int] = {}
    for c in prev_clusters:
        if c.centroid_embedding:
            prev_vecs.append(_normalize(np.array(c.centroid_embedding, dtype=float)))
            prev_sizes[c.id] = 0

    trends: List[Dict[str, Any]] = []
    
    # 2. Process Current Clusters
    for cluster in clusters:
        items = cluster.get("items", [])
        # Compute centroid if not already present
        centroid = cluster.get("centroid_embedding") or _centroid(items)
        cluster["centroid_embedding"] = centroid

        if not centroid:
            novelty = 1.0
            sim = 0.0
        else:
            # 3. Compare with History
            vec = _normalize(np.array(centroid, dtype=float))
            sim = _best_match_sim(vec, prev_vecs)
            novelty = max(0.0, min(1.0, 1.0 - sim))

        size = len(items)
        trend_type = _trend_type(novelty, size)
        confidence = _confidence(size, novelty)

        trend = {
            "cluster_label": cluster.get("label"),
            "cluster_id": cluster.get("id"),
            "trend_type": trend_type,
            "novelty_score": novelty,
            "confidence": confidence,
            "items": items,
            "best_prev_similarity": sim,
        }
        
        # 4. Boost score based on source diversity
        _apply_corroboration_factor(trend)

        trends.append(
            {
                **trend,
            }
        )

    return trends


def _apply_corroboration_factor(trend: Dict[str, Any]) -> None:
    """
    Boost item scores based on source diversity.
    
    Logic:
    - Count distinct sources (e.g., arxiv, github, blog).
    - Factor = 1.0 + 0.1 for each additional source beyond the first (max 0.5 boost).
    - Applied mainly for downstream ranking/display.
    """
    sources = {}
    for item in trend.get("items", []) or []:
        src = item.get("source") or "unknown"
        sources[src] = sources.get(src, 0) + 1
    distinct = len(sources)
    factor = 1.0 + min(0.5, 0.1 * max(0, distinct - 1))
    for item in trend.get("items", []) or []:
        item["corroboration_factor"] = factor
