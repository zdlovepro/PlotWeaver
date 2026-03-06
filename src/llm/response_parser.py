import json
import re
from typing import Any, Optional


def extract_json(text: str) -> Any:
    """Extract the first JSON object or array from a text string (LLM response)."""
    # Try to find JSON in code fences first
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try to parse the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find the outermost JSON object or array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"No valid JSON found in response:\n{text[:500]}")


def parse_plot_units(response_text: str) -> list[dict]:
    """Parse a list of plot units from an LLM response."""
    data = extract_json(response_text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "plots" in data:
        return data["plots"]
    raise ValueError(f"Expected a JSON array of plot units, got: {type(data)}")


def parse_annotation(response_text: str) -> dict:
    """Parse a single annotation dict from an LLM response."""
    data = extract_json(response_text)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Expected a JSON object for annotation, got: {type(data)}")


def parse_outline(response_text: str) -> dict:
    """Parse a novel outline dict from an LLM response."""
    data = extract_json(response_text)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Expected a JSON object for outline, got: {type(data)}")
