from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List


def render_report(week: str, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trends:
        trends = [
            {
                "title": "No trends computed",
                "summary": "Pipeline ran but did not emit any trend items.",
                "so_what": "Add sources and enable ingestion to populate trends.",
                "evidence": {
                    "source_counts": {},
                    "trust_score": 0.0,
                    "conflicts": 0,
                },
                "confidence": 0.0,
            }
        ]

    md_lines = [f"# Week {week} Summary", "", "## Trends"]
    for idx, t in enumerate(trends, start=1):
        md_lines.append(f"{idx}. {t.get('title', 'Untitled')}")
        md_lines.append(f"   - Summary: {t.get('summary', '')}")
        md_lines.append(f"   - So what: {t.get('so_what', '')}")
        evidence = t.get("evidence", {})
        md_lines.append(
            "   - Evidence: "
            f"sources={evidence.get('source_counts', {})}, "
            f"trust={evidence.get('trust_score', 0.0)}, "
            f"conflicts={evidence.get('conflicts', 0)}"
        )
        md_lines.append("")

    report_json = {
        "week": week,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trends": trends,
    }

    return {
        "markdown": "\n".join(md_lines).rstrip() + "\n",
        "json": json.dumps(report_json, indent=2),
    }
