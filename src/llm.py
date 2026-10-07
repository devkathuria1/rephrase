"""Single-call LLM explanation and rephrasing via the Anthropic API."""

from __future__ import annotations

import json
import os
import re

import config
from prompts import SYSTEM_PROMPT, build_user_message

KEYS = ("explanation", "rephrased", "intent_note", "confidence_caveat")


def is_enabled() -> bool:
    """Return True if an Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def parse_response(raw: str) -> dict[str, str]:
    """Parse and validate the model's JSON reply.

    Tolerates markdown code fences and surrounding prose.

    Raises:
        ValueError: if no JSON object is found, or required keys are missing
            or not strings.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    for key in KEYS:
        if not isinstance(data.get(key), str):
            raise ValueError(f"missing or non-string key: {key}")
    if not data["rephrased"].strip():
        raise ValueError("empty rephrasing")
    return {key: data[key].strip() for key in KEYS}


def rephrase(
    text: str,
    context: str,
    tier: str,
    scores: dict[str, float],
    top_words: list[str],
) -> dict:
    """Ask the LLM to explain and rephrase the text in one call.

    Returns:
        ``{"ok": True, **fields}`` on success, otherwise
        ``{"ok": False, "error": str}``. Never raises on API or parse errors.
    """
    import anthropic

    message = build_user_message(text, context, tier, scores, top_words)
    try:
        client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_SECONDS)
        response = client.messages.create(
            model=config.llm_model(),
            max_tokens=config.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        return {"ok": True, **parse_response(raw)}
    except anthropic.APIError as exc:
        return {"ok": False, "error": f"LLM API error: {type(exc).__name__}"}
    except ValueError as exc:
        return {"ok": False, "error": f"Could not parse LLM response: {exc}"}
