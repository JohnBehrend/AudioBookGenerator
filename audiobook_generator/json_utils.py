"""Tolerant JSON extraction for LLM output, shared across the pipeline.

LLMs return JSON wrapped in prose or markdown fences, with single quotes, or
stringified. These helpers extract a usable object from that mess. Modules
with output-shape-specific parsing (e.g. speaker maps with unquoted keys)
build on top of these primitives.
"""

import ast
import json
import re
from typing import Any, Dict, Optional


def strip_markdown_fences(text: str) -> str:
    """Remove ``` / ```json code fences, keeping their contents."""
    return re.sub(r"```(?:json)?\n?([\s\S]*?)\n?```", r"\1", text)


def extract_first_json_block(text: str) -> Optional[str]:
    """Extract the first balanced ``{...}`` block from text (fences stripped).

    Returns:
        The JSON substring, or None if no balanced object is found.
    """
    text = strip_markdown_fences(text)
    first = text.find("{")
    if first == -1:
        return None
    depth = 0
    for i in range(first, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[first:i + 1]
    return None


def extract_json_dict(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract a dict from LLM output, tolerating imperfect JSON.

    Tries, in order: strict ``json.loads`` on the first balanced ``{...}``
    block, then ``ast.literal_eval`` (fixes single-quoted / Python-style
    literals). Returns None if nothing parseable is found.
    """
    if not text:
        return None
    body = extract_first_json_block(text)
    if body is None:
        return None
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        # Single-quoted keys/values are invalid JSON but valid Python.
        obj = ast.literal_eval(body)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None
