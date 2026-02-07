from __future__ import annotations

from datetime import date


def week_id(d: date | None = None) -> str:
    d = d or date.today()
    year, week, _ = d.isocalendar()
    return f"{year}-{week:02d}"
