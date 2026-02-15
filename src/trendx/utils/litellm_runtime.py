from __future__ import annotations

import litellm

from trendx.config import LLMConfig
from trendx.utils.logging import get_logger

logger = get_logger(__name__)


def configure_litellm(llm_cfg: LLMConfig) -> None:
    """
    Apply process-wide LiteLLM runtime settings from TrendX config.
    """
    litellm.drop_params = bool(llm_cfg.drop_unsupported_params)
    logger.info("litellm.drop_params=%s", litellm.drop_params)
