from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_DEBUG_ENABLED = False
_DEBUG_PATH: Optional[Path] = None


def enable_debug(path: str | Path) -> None:
    global _DEBUG_ENABLED, _DEBUG_PATH
    _DEBUG_ENABLED = True
    _DEBUG_PATH = Path(path)
    _DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _truncate(value: Any, max_chars: int = 5000) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "\n...[truncated]"
    if isinstance(value, dict):
        return {k: _truncate(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, max_chars) for v in value]
    return value


def log_llm_event(
    stage: str,
    model: str,
    request: Dict[str, Any],
    response: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    if not _DEBUG_ENABLED or _DEBUG_PATH is None:
        return

    payload: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "model": model,
        "request": _truncate(request),
    }
    if response is not None:
        payload["response"] = _truncate(response)
    if extra:
        payload["extra"] = _truncate(extra)

    _DEBUG_PATH.write_text(
        (_DEBUG_PATH.read_text() if _DEBUG_PATH.exists() else "")
        + json.dumps(payload)
        + "\n",
        encoding="utf-8",
    )


def log_run_metadata(metadata: Dict[str, Any]) -> None:
    if not _DEBUG_ENABLED or _DEBUG_PATH is None:
        return
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "metadata": _truncate(metadata),
    }
    line = "----- [metadata]---- " + json.dumps(payload) + "\n"
    _DEBUG_PATH.write_text(
        (_DEBUG_PATH.read_text() if _DEBUG_PATH.exists() else "") + line,
        encoding="utf-8",
    )
