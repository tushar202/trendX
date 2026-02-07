from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from jsonschema import Draft202012Validator

from trendx.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_MAP = {
    "discover": "schemas/discover.schema.json",
    "ingest": "schemas/ingest.schema.json",
    "normalize": "schemas/normalize.schema.json",
    "filter": "schemas/filter.schema.json",
    "enrich": "schemas/enrich.schema.json",
    "dedupe": "schemas/dedupe.schema.json",
    "cluster": "schemas/cluster.schema.json",
    "trend": "schemas/trend.schema.json",
    "critique": "schemas/critique.schema.json",
    "synthesis": "schemas/synthesis.schema.json",
    "report": "schemas/report.schema.json",
}


def _load_schema(schema_path: str) -> Dict[str, Any] | None:
    path = Path(schema_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def validate_payload(stage: str, payload: Dict[str, Any]) -> Tuple[bool, list[str]]:
    schema_path = SCHEMA_MAP.get(stage)
    if not schema_path:
        return True, []

    schema = _load_schema(schema_path)
    if not schema:
        logger.warning("schema missing for stage=%s", stage)
        return True, []

    validator = Draft202012Validator(schema)
    errors = [e.message for e in validator.iter_errors(payload)]
    if errors:
        logger.error("validation failed stage=%s errors=%s", stage, errors)
        return False, errors

    return True, []
