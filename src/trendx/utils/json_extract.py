from __future__ import annotations

import re
from typing import Any, Optional


def _strip_fences(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _coerce_json_literals(text: str) -> str:
    text = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
    return text


def _literal_eval_fallback(text: str) -> Any | None:
    try:
        import ast

        return ast.literal_eval(text)
    except Exception:
        return None


def _find_json_slice(text: str) -> Optional[str]:
    # Find first { or [
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None

    # Brace matching
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                return None
            last = stack.pop()
            if (last == "{" and ch != "}") or (last == "[" and ch != "]"):
                return None
            if not stack:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any | None:
    text = _strip_fences(text)
    try:
        import json

        return json.loads(text)
    except Exception:
        pass

    snippet = _find_json_slice(text)
    if not snippet:
        return None
    try:
        import json

        return json.loads(snippet)
    except Exception:
        cleaned = _remove_trailing_commas(snippet)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        python_like = _coerce_json_literals(cleaned)
        return _literal_eval_fallback(python_like)
