"""Single-call LLM explanation and rephrasing through a pluggable provider.

Providers (``LLM_PROVIDER``): ``groq``, ``ollama``, ``anthropic`` or ``none``.
API keys are read from the environment and never logged or returned.
"""

from __future__ import annotations

import json
import os
import re

import requests

import config
from prompts import SYSTEM_PROMPT, build_user_message

KEYS = ("explanation", "rephrased", "intent_note", "confidence_caveat")
_KEY_ENV = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class ProviderError(Exception):
    """A provider call failed; the message is safe to show (it never holds secrets)."""


def disabled_reason() -> str | None:
    """Return why rephrasing is unavailable, or None if it can be attempted."""
    provider = config.llm_provider()
    if provider == "none":
        return "LLM_PROVIDER is 'none'."
    if provider not in config.PROVIDERS:
        return f"Unknown LLM_PROVIDER '{provider}' (expected one of {', '.join(config.PROVIDERS)})."
    key_env = _KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env):
        return f"{key_env} is not set."
    return None


def is_enabled() -> bool:
    """Return True if the configured provider can be attempted."""
    return disabled_reason() is None


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


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    """POST JSON and return the decoded body, mapping failures to ProviderError."""
    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        raise ProviderError(f"network error ({type(exc).__name__})") from None
    if resp.status_code == 429:
        raise ProviderError("rate limited (HTTP 429)")
    if resp.status_code != 200:
        raise ProviderError(f"HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        raise ProviderError("response body was not JSON") from None


def _call_groq(system: str, user: str) -> str:
    """Call Groq's OpenAI-compatible chat endpoint in JSON mode."""
    body = _post_json(
        config.GROQ_URL,
        {
            "model": config.llm_model("groq"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": 0.3,
        },
        {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    )
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError("unexpected response shape") from None


def _call_ollama(system: str, user: str) -> str:
    """Call a local Ollama server's chat endpoint in JSON mode."""
    body = _post_json(
        f"{config.ollama_url()}/api/chat",
        {
            "model": config.llm_model("ollama"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
        },
    )
    try:
        return body["message"]["content"]
    except (KeyError, TypeError):
        raise ProviderError("unexpected response shape") from None


def _call_anthropic(system: str, user: str) -> str:
    """Call the Anthropic Messages API."""
    import anthropic

    try:
        client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_SECONDS)
        response = client.messages.create(
            model=config.llm_model("anthropic"),
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as exc:
        raise ProviderError(f"Anthropic API error ({type(exc).__name__})") from None
    return "".join(b.text for b in response.content if b.type == "text")


_PROVIDER_CALLS = {"groq": _call_groq, "ollama": _call_ollama, "anthropic": _call_anthropic}


def rephrase(
    text: str,
    context: str,
    tier: str,
    scores: dict[str, float],
    top_words: list[str],
) -> dict:
    """Ask the configured provider to explain and rephrase the text in one call.

    Returns:
        ``{"ok": True, **fields}`` on success, otherwise
        ``{"ok": False, "error": str}``. Never raises on provider or parse errors.
    """
    reason = disabled_reason()
    if reason:
        return {"ok": False, "error": f"Rephrasing disabled: {reason}"}
    provider = config.llm_provider()
    user = build_user_message(text, context, tier, scores, top_words)
    try:
        return {"ok": True, **parse_response(_PROVIDER_CALLS[provider](SYSTEM_PROMPT, user))}
    except ProviderError as exc:
        return {"ok": False, "error": f"Rephrasing disabled: {provider} failed: {exc}"}
    except ValueError as exc:
        return {"ok": False, "error": f"Rephrasing disabled: {provider} returned unusable output: {exc}"}
