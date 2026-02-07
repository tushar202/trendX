from __future__ import annotations

from typing import Any, Dict, List

from trendx.formatter.render import render_report


def run(week: str, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
    return render_report(week, trends)
