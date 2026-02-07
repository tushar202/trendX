from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Source(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    type: str
    name: str
    base_url: Optional[str] = None
    trust_weight: float = 1.0


class Item(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: Optional[int] = Field(default=None, foreign_key="source.id")
    url: str
    canonical_url: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    published_at: Optional[datetime] = None
    raw_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    hash: Optional[str] = None
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))
    trust_score: Optional[float] = None


class RepoMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id")
    stars: Optional[int] = None
    forks: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    license: Optional[str] = None
    has_tests: bool = False
    has_dockerfile: bool = False
    has_docker_compose: bool = False


class Claim(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id")
    claim_text: str
    evidence_score: Optional[float] = None
    novelty_score: Optional[float] = None


class EvidenceAnchor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: int = Field(foreign_key="claim.id")
    item_id: int = Field(foreign_key="item.id")
    anchor_text: str
    anchor_type: str
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    confidence: Optional[float] = None


class Cluster(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    week: str
    label: str
    centroid_embedding: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))
    summary: Optional[str] = None


class ClusterItem(SQLModel, table=True):
    cluster_id: int = Field(foreign_key="cluster.id", primary_key=True)
    item_id: int = Field(foreign_key="item.id", primary_key=True)


class Trend(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    week: str
    cluster_id: int = Field(foreign_key="cluster.id")
    trend_type: str
    novelty_score: Optional[float] = None
    confidence: Optional[float] = None


class Report(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    week: str
    json_path: str
    md_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stage: str
    item_id: Optional[int] = None
    status: str
    errors: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
