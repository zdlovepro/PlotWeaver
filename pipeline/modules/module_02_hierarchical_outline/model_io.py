"""Checkpointed JSON model calls owned by module 02."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ...common.jsonio import read_json, write_json
from ...common.model import ModelSettings, complete_json, json_system_message
from . import MODULE_VERSION
from .prompts import OUTLINE_SYSTEM_PROMPT
from .request_budget import OutlineRequestLedger


JsonCompletion = Callable[..., dict[str, Any]]


def _fingerprint(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class CheckpointedOutlineModel:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        source_fingerprint: str,
        settings: ModelSettings,
        resume: bool,
        ledger: OutlineRequestLedger,
        completion: JsonCompletion = complete_json,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.source_fingerprint = source_fingerprint
        self.settings = settings
        self.resume = resume
        self.ledger = ledger
        self.completion = completion
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def call(
        self,
        *,
        stage: str,
        identity: str,
        revision: str,
        prompt: str,
        max_tokens: int,
        thinking: bool = False,
    ) -> dict[str, Any]:
        effective_system = json_system_message(OUTLINE_SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]
        request = {
            "base_system": OUTLINE_SYSTEM_PROMPT,
            "system": effective_system,
            "user": prompt,
            "messages": messages,
            "model": self.settings.model,
            "base_url": self.settings.base_url,
            "thinking": thinking,
            "max_tokens": max_tokens,
        }
        request["request_fingerprint"] = _fingerprint(request)
        input_chars = len(effective_system) + len(prompt)
        self.ledger.reserve(stage, input_chars)
        checkpoint_key = _fingerprint({
            "source": self.source_fingerprint,
            "module_version": MODULE_VERSION,
            "revision": revision,
            "stage": stage,
            "identity": identity,
            "request": request["request_fingerprint"],
        })
        path = self.checkpoint_dir / f"{checkpoint_key}.{stage}.json"
        if self.resume:
            cached = read_json(path)
            if (
                isinstance(cached, dict)
                and cached.get("stage") == stage
                and isinstance(cached.get("response"), dict)
            ):
                self.ledger.mark_cache_hit()
                return cached["response"]

        write_json(path.with_suffix(".request.json"), {
            "stage": stage,
            "identity": identity,
            "module_version": MODULE_VERSION,
            "prompt_revision": revision,
            "source_fingerprint": self.source_fingerprint,
            "request": request,
        })
        path.with_suffix(".request.txt").write_text(
            f"[SYSTEM]\n{effective_system}\n\n[USER]\n{prompt}\n",
            encoding="utf-8",
        )

        def observe(
            attempt: int,
            content: str,
            parse_error: str | None,
            metadata: dict[str, Any],
        ) -> None:
            path.with_suffix(f".attempt-{attempt:02d}.raw.txt").write_text(
                content, encoding="utf-8"
            )
            write_json(path.with_suffix(f".attempt-{attempt:02d}.meta.json"), {
                "stage": stage,
                "attempt": attempt,
                "model": self.settings.model,
                "input_character_count": input_chars,
                "output_character_count": len(content),
                "json_parse_succeeded": parse_error is None,
                "parse_error": parse_error,
                "finish_reason": metadata.get("finish_reason"),
                "usage": metadata.get("usage", {}),
            })

        self.ledger.mark_actual_call(stage, input_chars)
        kwargs: dict[str, Any] = {
            "attempts": 1,
            "max_tokens": max_tokens,
            "thinking": thinking,
            "allow_thinking_fallback": False,
        }
        if self.completion is complete_json:
            kwargs["attempt_observer"] = observe
        try:
            response = self.completion(
                OUTLINE_SYSTEM_PROMPT,
                prompt,
                self.settings,
                **kwargs,
            )
        except Exception as exc:
            setattr(exc, "failed_stage", stage)
            raise
        if not isinstance(response, dict):
            error = TypeError("outline model completion must return a JSON object")
            setattr(error, "failed_stage", stage)
            raise error
        write_json(path, {
            "stage": stage,
            "identity": identity,
            "module_version": MODULE_VERSION,
            "prompt_revision": revision,
            "source_fingerprint": self.source_fingerprint,
            "request": request,
            "response": response,
        })
        return response
