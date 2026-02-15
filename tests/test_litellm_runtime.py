from __future__ import annotations

from pathlib import Path

import litellm

from trendx.config import load_config
from trendx.utils.litellm_runtime import configure_litellm


def test_load_config_maps_legacy_litellm_drop_params(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openai",
                "  model: gpt-5-nano",
                "  litellm.drop_params: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.llm.drop_unsupported_params is True


def test_configure_litellm_applies_drop_params_from_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openai",
                "  model: gpt-5-nano",
                "  drop_unsupported_params: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    original = litellm.drop_params
    try:
        configure_litellm(cfg.llm)
        assert litellm.drop_params is False
    finally:
        litellm.drop_params = original
