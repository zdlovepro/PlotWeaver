from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _file_settings() -> dict[str, Any]:
    # pipeline/<module>.py -> workspace root is one directory above ``pipeline``.
    root = Path(__file__).resolve().parents[1]
    for name in ("config.local.yaml", "config.yaml"):
        path = root / name
        if not path.exists():
            continue
        try:
            import yaml
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return payload.get("deepseek", {}) if isinstance(payload, dict) else {}
        except Exception:
            continue
    return {}


@dataclass(frozen=True)
class ModelSettings:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 6000
    # 章节流水线由多次受限调用组成。单次长时间卡住会阻塞整个批次，
    # 因此把超时限定在可诊断的范围；可由 config.yaml 或环境变量覆盖。
    request_timeout: float = 75.0

    @classmethod
    def from_environment(cls) -> "ModelSettings":
        file_settings = _file_settings()
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or str(file_settings.get("api_key", "")),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "") or str(file_settings.get("base_url", "https://api.deepseek.com")),
            model=os.environ.get("DEEPSEEK_MODEL", "") or str(file_settings.get("model", "deepseek-chat")),
            max_tokens=int(os.environ.get("DISTILL_MAX_TOKENS", "") or file_settings.get("max_tokens", 6000)),
            request_timeout=float(os.environ.get("DISTILL_REQUEST_TIMEOUT", "") or file_settings.get("request_timeout", 75)),
        )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def _response_json_or_error(content: str | None) -> dict[str, Any]:
    """Keep an absent provider message distinct from an empty JSON object."""

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型返回空内容，未产生 JSON object")
    return extract_json(content)


def complete_json(system: str, user: str, settings: ModelSettings, attempts: int = 2, max_tokens: int | None = None) -> dict[str, Any]:
    if not settings.api_key:
        raise RuntimeError("No model API key found. Set DEEPSEEK_API_KEY or use --offline.")
    from openai import OpenAI
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.request_timeout, max_retries=0)
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": f"{system}\n\n输出契约：你的全部回复必须是一个符合 RFC 8259 的合法 JSON object。不得输出 Markdown、代码块、解释、前缀或后缀。JSON 字符串中的换行和引号必须正确转义。"},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens or settings.max_tokens,
            )
            return _response_json_or_error(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 8))
    detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
    raise RuntimeError(f"Model did not return valid JSON after {max(1, attempts)} attempt(s){detail}") from last_error


def complete_text(system: str, user: str, settings: ModelSettings, attempts: int = 2, max_tokens: int | None = None) -> str:
    """Call the model for prose-only output without a JSON response contract."""
    if not settings.api_key:
        raise RuntimeError("No model API key found. Set DEEPSEEK_API_KEY or use --offline.")
    from openai import OpenAI
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.request_timeout, max_retries=0)
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens or settings.max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model text request failed after {max(1, attempts)} attempt(s)") from last_error
