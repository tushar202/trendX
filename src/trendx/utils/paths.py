from __future__ import annotations

from datetime import datetime
from pathlib import Path


def get_new_run_dir(week: str) -> Path:
    run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    return Path("reports") / week / run_id
