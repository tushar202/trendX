from __future__ import annotations

from typing import Any, Dict, List

from trendx.trend.detect import detect_trends


async def run(db_url: str, week: str, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return await detect_trends(db_url, week, clusters)
