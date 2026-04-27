"""
utils.py – Shared utilities for PlotWeaver pipeline.

Provides:
  - get_deepseek_client()  – returns an openai.OpenAI client pointed at DeepSeek.
  - get_chromadb_client()  – returns a chromadb.Client based on config.
  - chat_completion_json() – thin wrapper for a single chat call.
"""

from __future__ import annotations

from typing import Optional

import config


def get_deepseek_client():
    """Return an OpenAI-compatible client configured for DeepSeek."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        ) from exc

    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )


def get_chromadb_client():
    """Return a ChromaDB client based on config."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb package is required. Install with: pip install chromadb"
        ) from exc

    if config.CHROMADB_MODE == "http":
        return chromadb.HttpClient(
            host=config.CHROMADB_HOST, port=config.CHROMADB_PORT
        )
    # Default: local persistent client
    return chromadb.PersistentClient(path=config.CHROMADB_PATH)


def _supports_reasoning_mode(model_name: str) -> bool:
    return str(model_name or "").startswith("deepseek-v4")


def _supports_temperature(model_name: str) -> bool:
    return not _supports_reasoning_mode(model_name)


def chat_completion_json(
    client,
    system: str,
    user: str,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Call the DeepSeek chat completion API and return the assistant message content.

    Args:
        client:      OpenAI-compatible client.
        system:      System prompt.
        user:        User prompt.
        json_mode:   If True, set response_format to json_object.
        max_tokens:  Override config default.
        temperature: Override config default.

    Returns:
        String content of the assistant reply.
    """
    model_name = model_name or config.DEEPSEEK_MODEL
    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens or config.DEEPSEEK_MAX_TOKENS,
    }
    if _supports_temperature(model_name):
        kwargs["temperature"] = temperature if temperature is not None else config.DEEPSEEK_TEMPERATURE
    if _supports_reasoning_mode(model_name):
        kwargs["reasoning_effort"] = config.DEEPSEEK_REASONING_EFFORT
        if config.DEEPSEEK_THINKING_ENABLED:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
