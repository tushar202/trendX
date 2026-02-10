from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel


class TrendDraft(BaseModel):
    id: str  # stable id: f"{week}:{cluster_label}"
    cluster_label: str
    title: str
    summary: str
    so_what: str
    status: Literal["draft", "rejected", "approved"] = "draft"
    feedback_history: List[str] = []
    revision_count: int = 0
    needs_review: bool = False  # True if auto-approved after hitting max revisions


class ReportState(BaseModel):
    week: str
    run_dir: str
    step: Literal[
        "start",
        "ingest",
        "cluster",
        "synthesis",
        "audit",
        "publish",
        "done",
    ] = "start"
    items_count: int = 0
    cluster_count: int = 0
    drafts: List[TrendDraft] = []

    def save(self) -> None:
        base = Path(self.run_dir)
        base.mkdir(parents=True, exist_ok=True)
        (base / "state.json").write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, run_dir: str):
        path = Path(run_dir) / "state.json"
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return None
