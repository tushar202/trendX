from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    drop_unsupported_params: bool = True
    debug: bool = False
    debug_path: Optional[str] = None
    tavily_api_key: Optional[str] = None


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class StorageConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///./trendx.db"


class RunConfig(BaseModel):
    cadence: str = "weekly"
    max_items_per_source: int = 200


class PapersConfig(BaseModel):
    enabled: bool = True
    arxiv: Dict[str, Any] = Field(default_factory=dict)
    semantic_scholar: Dict[str, Any] = Field(default_factory=dict)


class BlogsConfig(BaseModel):
    enabled: bool = True
    rss_feeds: List[str] = Field(default_factory=list)


class GitHubConfig(BaseModel):
    enabled: bool = True
    trending: bool = True
    releases: bool = True
    starred_updates: bool = False
    search_query: str = "agentic ai"


class SocialConfig(BaseModel):
    enabled: bool = True
    hacker_news: bool = True
    reddit: Dict[str, Any] = Field(default_factory=dict)


class SourcesConfig(BaseModel):
    papers: PapersConfig = Field(default_factory=PapersConfig)
    blogs: BlogsConfig = Field(default_factory=BlogsConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    social: SocialConfig = Field(default_factory=SocialConfig)


class TrustConfig(BaseModel):
    source_weights: Dict[str, float] = Field(default_factory=dict)


class AgenticSynthesisConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable critique-refine loop")
    retrieval_enabled: bool = Field(
        default=True, description="Enable local evidence search"
    )
    max_retries: int = Field(default=2, description="Max refinement attempts per trend")
    quality_threshold: int = Field(default=8, description="Score (0-10) to accept draft")
    critic_model: Optional[str] = Field(
        default=None, description="Optional model for critique"
    )


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    trust: TrustConfig = Field(default_factory=TrustConfig)
    agentic_synthesis: AgenticSynthesisConfig = Field(
        default_factory=AgenticSynthesisConfig
    )


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    llm = data.get("llm")
    if isinstance(llm, dict):
        # Backward compatibility for configs using dotted key syntax.
        if "litellm.drop_params" in llm and "drop_unsupported_params" not in llm:
            llm["drop_unsupported_params"] = bool(llm.get("litellm.drop_params"))
    return AppConfig.model_validate(data)
