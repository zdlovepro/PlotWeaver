from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest


class ModelServiceError(RuntimeError):
    """The provider could not execute a request, independent of content quality."""


class ModelOutputError(RuntimeError):
    """The provider responded, but no valid result satisfied the JSON contract."""


JSON_OUTPUT_CONTRACT = (
    "输出契约：你的全部回复必须是一个符合 RFC 8259 的合法 JSON object。"
    "不得输出 Markdown、代码块、解释、前缀或后缀。JSON 字符串中的换行和引号必须正确转义。"
)


def json_system_message(system: str) -> str:
    """Return the exact system content used by complete_json."""

    return f"{system}\n\n{JSON_OUTPUT_CONTRACT}"


def _thinking_fallback_allowed(error: ModelServiceError) -> bool:
    """Return whether disabling thinking could plausibly recover the request.

    Unsupported request fields and thinking-time timeouts may be mode-specific.
    Authentication, balance, permission and rate-limit responses are not; a
    second identical billable request only delays a resumable pipeline.
    """

    detail = str(error).lower()
    terminal_markers = (
        "http 401", "http 402", "http 403", "http 429",
        "insufficient balance", "余额不足", "欠费", "quota", "rate limit",
        "invalid api key", "authentication", "permission denied",
    )
    return not any(marker in detail for marker in terminal_markers)


def _strip_yaml_comment(value: str) -> str:
    """Remove an unquoted YAML comment from the small local config subset."""

    quote = ""
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
        elif char == "#" and not quote:
            return value[:index].rstrip()
    return value.strip()


def _yaml_scalar(value: str) -> Any:
    value = _strip_yaml_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _minimal_deepseek_yaml(text: str) -> dict[str, Any]:
    """Read the flat ``deepseek:`` section when PyYAML is unavailable.

    This intentionally supports only the repository's checked-in configuration
    shape: a top-level section with scalar children.  It avoids turning model
    availability into an optional dependency while leaving full YAML parsing to
    PyYAML whenever it is installed.
    """

    values: dict[str, Any] = {}
    in_section = False
    section_indent = 0
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if not in_section:
            if line == "deepseek:":
                in_section = True
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key and value.strip():
            values[key] = _yaml_scalar(value)
    return values


def _file_settings() -> dict[str, Any]:
    # pipeline/common/<module>.py -> workspace root is two directories above.
    root = Path(__file__).resolve().parents[2]
    for name in ("config.local.yaml", "config.yaml"):
        path = root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            import yaml
            payload = yaml.safe_load(text) or {}
            return payload.get("deepseek", {}) if isinstance(payload, dict) else {}
        except ImportError:
            return _minimal_deepseek_yaml(text)
        except Exception:
            # A malformed full YAML file should not be interpreted loosely.
            # Continue to a lower-priority config file or environment values.
            continue
    return {}


@dataclass(frozen=True)
class ModelSettings:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    review_model: str = ""
    max_tokens: int = 6000
    reasoning_effort: str = "high"
    thinking_enabled: bool = True
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
            review_model=(
                os.environ.get("DISTILL_REVIEW_MODEL", "")
                or str(file_settings.get("review_model", ""))
                or str(file_settings.get("last_steps_model", ""))
            ),
            max_tokens=int(os.environ.get("DISTILL_MAX_TOKENS", "") or file_settings.get("max_tokens", 6000)),
            reasoning_effort=str(
                os.environ.get("DEEPSEEK_REASONING_EFFORT", "")
                or file_settings.get("reasoning_effort", "high")
            ),
            thinking_enabled=str(
                os.environ.get("DEEPSEEK_THINKING_ENABLED", "")
                or file_settings.get("thinking_enabled", True)
            ).strip().lower() not in {"0", "false", "no", "off", "disabled"},
            request_timeout=float(os.environ.get("DISTILL_REQUEST_TIMEOUT", "") or file_settings.get("request_timeout", 75)),
        )

    def for_review(self) -> "ModelSettings":
        """Return settings for an independent high-value review pass."""

        return replace(self, model=self.review_model or self.model)

    def for_quality(self) -> "ModelSettings":
        """Use the configured high-quality model for content-critical stages.

        ``review_model`` is intentionally reused as the quality tier because
        existing configuration already exposes it as ``last_steps_model``.
        Callers may still inject explicit settings in tests or local runs.
        """

        return self.for_review()


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


def _compat_chat_completion(
    settings: ModelSettings,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    json_object: bool,
    thinking_mode: str | None = None,
    reasoning_effort: str | None = None,
    request_timeout: float | None = None,
    metadata_observer: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint without an SDK dependency."""

    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    if thinking_mode is not None:
        payload["thinking"] = {"type": thinking_mode}
    if thinking_mode == "enabled" and reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = f"{settings.base_url.rstrip('/')}/chat/completions"
    request = urlrequest.Request(
        endpoint,
        data=body,
        headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=request_timeout or settings.request_timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        # Provider responses are useful for a malformed request, but never
        # include request headers or secrets in the diagnostic.
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ModelServiceError(f"模型服务返回 HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise ModelServiceError(f"模型服务连接失败: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ModelServiceError(
            f"模型服务请求超过 {request_timeout or settings.request_timeout:g} 秒"
        ) from exc
    if not isinstance(decoded, dict):
        raise ModelServiceError("模型服务返回的顶层 JSON 不是对象")
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelServiceError("模型服务响应缺少 choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelServiceError("模型服务响应缺少 message")
    content = message.get("content")
    if metadata_observer is not None:
        usage = decoded.get("usage")
        reasoning_content = str(message.get("reasoning_content") or "")
        metadata_observer({
            "finish_reason": choice.get("finish_reason"),
            "usage": usage if isinstance(usage, dict) else {},
            "reasoning_content_character_count": len(reasoning_content),
            "reasoning_content": reasoning_content,
        })
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ModelServiceError("模型服务响应缺少文本 content")
    return content


def _sdk_or_compat_completion(
    settings: ModelSettings,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    json_object: bool,
    thinking_mode: str | None = None,
    metadata_observer: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Use the bounded protocol client, with SDK available only by opt-in.

    Some OpenAI-compatible SDK stacks do not enforce a strict wall around a
    request on all Windows/proxy combinations.  This pipeline makes many small
    calls, so one unbounded SDK request can stall an entire book.  The standard
    compatible endpoint is therefore the default; ``DISTILL_MODEL_TRANSPORT``
    may explicitly select ``sdk`` for environments that have verified it.
    """

    # Deep reasoning endpoints regularly need longer before the first response
    # byte than bounded non-thinking JSON calls.  Keep the configured timeout
    # for ordinary calls and apply a finite multiplier only to an explicitly
    # selected thinking task.
    request_timeout = settings.request_timeout * (3 if thinking_mode == "enabled" else 1)
    transport = os.environ.get("DISTILL_MODEL_TRANSPORT", "compat").strip().lower()
    if transport != "sdk":
        return _compat_chat_completion(
            settings,
            messages,
            max_tokens=max_tokens,
            json_object=json_object,
            thinking_mode=thinking_mode,
            reasoning_effort=settings.reasoning_effort,
            request_timeout=request_timeout,
            metadata_observer=metadata_observer,
        )
    try:
        from openai import OpenAI
    except ImportError:
        return _compat_chat_completion(
            settings,
            messages,
            max_tokens=max_tokens,
            json_object=json_object,
            thinking_mode=thinking_mode,
            reasoning_effort=settings.reasoning_effort,
            request_timeout=request_timeout,
            metadata_observer=metadata_observer,
        )
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=request_timeout, max_retries=0)
    kwargs: dict[str, Any] = {"model": settings.model, "messages": messages, "max_tokens": max_tokens}
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if thinking_mode is not None:
        kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
    if thinking_mode == "enabled" and settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # SDK exception classes differ across optional client versions.  Keep
        # the pipeline API stable and retain the redacted provider message.
        raise ModelServiceError(f"模型服务请求失败: {type(exc).__name__}: {exc}") from exc
    choice = response.choices[0]
    if metadata_observer is not None:
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif not isinstance(usage, dict):
            usage = {}
        reasoning_content = str(getattr(choice.message, "reasoning_content", "") or "")
        metadata_observer({
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage,
            "reasoning_content_character_count": len(str(reasoning_content)),
            "reasoning_content": reasoning_content,
        })
    return choice.message.content or ""


def _bounded_completion(
    settings: ModelSettings,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    json_object: bool,
    thinking_mode: str | None = None,
    metadata_observer: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Enforce a true wall-clock bound around either HTTP transport.

    Socket timeouts measure inactivity, not total elapsed time.  A local proxy
    or provider can keep a connection technically active while never closing
    the JSON response, which used to stall a whole work indefinitely.  The
    transport call therefore runs on a daemon worker and the pipeline stops
    waiting at the same finite budget advertised by ``ModelSettings``.  The
    daemon is intentionally unable to mutate pipeline state; only its returned
    string can cross this boundary.
    """

    wall_timeout = settings.request_timeout * (3 if thinking_mode == "enabled" else 1)
    completed = threading.Event()
    result: list[str] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            result.append(_sdk_or_compat_completion(
                settings,
                messages,
                max_tokens=max_tokens,
                json_object=json_object,
                thinking_mode=thinking_mode,
                metadata_observer=metadata_observer,
            ))
        except BaseException as exc:  # propagate the original typed error
            failures.append(exc)
        finally:
            completed.set()

    threading.Thread(target=invoke, name="plotweaver-model-request", daemon=True).start()
    if not completed.wait(wall_timeout):
        raise ModelServiceError(f"模型服务请求超过总墙钟 {wall_timeout:g} 秒")
    if failures:
        raise failures[0]
    if not result:
        raise ModelServiceError("模型服务请求结束但没有返回内容")
    return result[0]


def complete_json(
    system: str,
    user: str,
    settings: ModelSettings,
    attempts: int = 2,
    max_tokens: int | None = None,
    *,
    thinking: bool = False,
    allow_thinking_fallback: bool = True,
    attempt_observer: Callable[[int, str, str | None, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not settings.api_key:
        raise RuntimeError("No model API key found. Set DEEPSEEK_API_KEY or use --offline.")
    last_error: Exception | None = None

    def parse_observed(
        attempt_number: int,
        content: str,
        response_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = _response_json_or_error(content)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            if response_metadata.get("finish_reason") == "length":
                parse_error = f"ModelOutputTruncated: 模型输出达到max_tokens，JSON被截断；{parse_error}"
            if attempt_observer is not None:
                attempt_observer(attempt_number, content, parse_error, response_metadata)
            if response_metadata.get("finish_reason") == "length":
                raise ModelOutputError("模型输出达到max_tokens，JSON被截断") from exc
            raise
        if attempt_observer is not None:
            attempt_observer(attempt_number, content, None, response_metadata)
        return result

    for attempt in range(max(1, attempts)):
        try:
            messages = [
                {"role": "system", "content": json_system_message(system)},
                {"role": "user", "content": user},
            ]
            thinking_mode = "enabled" if thinking and settings.thinking_enabled else "disabled"
            response_metadata: dict[str, Any] = {}
            observe_metadata = lambda value: (response_metadata.clear(), response_metadata.update(value))
            try:
                content = _bounded_completion(
                    settings,
                    messages,
                    max_tokens=max_tokens or settings.max_tokens,
                    json_object=True,
                    thinking_mode=thinking_mode,
                    metadata_observer=observe_metadata,
                )
            except ModelServiceError as thinking_error:
                if (
                    thinking_mode != "enabled"
                    or not allow_thinking_fallback
                    or not _thinking_fallback_allowed(thinking_error)
                ):
                    raise
                # Thinking is a quality route, not a single point of failure.
                # A compatible endpoint may time out before returning public
                # content even though the same bounded task is serviceable.
                content = _bounded_completion(
                    settings,
                    messages,
                    max_tokens=max_tokens or settings.max_tokens,
                    json_object=True,
                    thinking_mode="disabled",
                    metadata_observer=observe_metadata,
                )
            try:
                return parse_observed(attempt + 1, content, response_metadata)
            except RuntimeError as empty_error:
                if (
                    thinking_mode != "enabled"
                    or not allow_thinking_fallback
                    or "空内容" not in str(empty_error)
                ):
                    raise
                # Some compatible endpoints spend the completion on private
                # reasoning and return an empty public content field.  Replay
                # the exact bounded task once without thinking; the prompt and
                # all downstream semantic gates remain unchanged.
                content = _bounded_completion(
                    settings,
                    messages,
                    max_tokens=max_tokens or settings.max_tokens,
                    json_object=True,
                    thinking_mode="disabled",
                    metadata_observer=observe_metadata,
                )
                return parse_observed(attempt + 1, content, response_metadata)
        except ModelServiceError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 8))
    detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
    raise ModelOutputError(
        f"Model did not return valid JSON after {max(1, attempts)} attempt(s){detail}"
    ) from last_error


def complete_text(system: str, user: str, settings: ModelSettings, attempts: int = 2, max_tokens: int | None = None) -> str:
    """Call the model for prose-only output without a JSON response contract."""
    if not settings.api_key:
        raise RuntimeError("No model API key found. Set DEEPSEEK_API_KEY or use --offline.")
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return _bounded_completion(
                settings,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens or settings.max_tokens,
                json_object=False,
            )
        except ModelServiceError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model text request failed after {max(1, attempts)} attempt(s)") from last_error
