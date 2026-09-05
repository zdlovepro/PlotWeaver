from __future__ import annotations

import unittest
import json
import time
from unittest.mock import patch

from pipeline.model import (
    ModelOutputError,
    ModelServiceError,
    ModelSettings,
    _compat_chat_completion,
    _bounded_completion,
    _sdk_or_compat_completion,
    _minimal_deepseek_yaml,
    complete_json,
)


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"{\\\"ok\\\":true}"}}]}'


class ModelConfigTests(unittest.TestCase):
    def test_dependency_free_reader_loads_the_flat_deepseek_section(self) -> None:
        payload = _minimal_deepseek_yaml(
            """# local config\ndeepseek:\n  api_key: \"placeholder\" # hidden\n  base_url: https://example.invalid\n  model: model-name\n  max_tokens: 32000\n  thinking_enabled: true\npaths:\n  input_dir: ./input\n"""
        )
        self.assertEqual(payload["api_key"], "placeholder")
        self.assertEqual(payload["base_url"], "https://example.invalid")
        self.assertEqual(payload["max_tokens"], 32000)
        self.assertTrue(payload["thinking_enabled"])
        self.assertNotIn("input_dir", payload)

    def test_review_settings_use_the_configured_stronger_model(self) -> None:
        settings = ModelSettings(model="fast-model", review_model="review-model")
        reviewer = settings.for_review()
        self.assertEqual(reviewer.model, "review-model")
        self.assertEqual(reviewer.review_model, "review-model")
        self.assertEqual(settings.for_quality().model, "review-model")

    def test_standard_library_compat_client_uses_openai_shape(self) -> None:
        settings = ModelSettings(api_key="placeholder", base_url="https://example.invalid", model="demo", max_tokens=16)
        with patch("pipeline.model.urlrequest.urlopen", return_value=_Response()) as request:
            content = _compat_chat_completion(
                settings,
                [{"role": "user", "content": "测试"}],
                max_tokens=12,
                json_object=True,
                thinking_mode="disabled",
            )
        self.assertEqual(content, '{"ok":true}')
        sent = request.call_args.args[0]
        self.assertTrue(sent.full_url.endswith("/chat/completions"))
        self.assertEqual(sent.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(sent.data.decode("utf-8"))["thinking"], {"type": "disabled"})
        self.assertEqual(request.call_args.kwargs["timeout"], settings.request_timeout)

    def test_standard_library_compat_client_carries_reasoning_controls(self) -> None:
        settings = ModelSettings(
            api_key="placeholder", base_url="https://example.invalid", model="demo",
            max_tokens=16, reasoning_effort="high",
        )
        with patch("pipeline.model.urlrequest.urlopen", return_value=_Response()) as request:
            _compat_chat_completion(
                settings,
                [{"role": "user", "content": "测试"}],
                max_tokens=12,
                json_object=True,
                thinking_mode="enabled",
                reasoning_effort=settings.reasoning_effort,
            )
        payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(request.call_args.kwargs["timeout"], settings.request_timeout)

    def test_complete_json_enables_thinking_only_when_requested(self) -> None:
        settings = ModelSettings(api_key="placeholder", thinking_enabled=True)
        with patch(
            "pipeline.model._sdk_or_compat_completion",
            return_value='{"ok":true}',
        ) as completion:
            complete_json("系统", "任务", settings, attempts=1, thinking=True)
        self.assertEqual(completion.call_args.kwargs["thinking_mode"], "enabled")

    def test_bounded_compat_transport_is_the_default_even_when_sdk_is_installed(self) -> None:
        settings = ModelSettings(api_key="placeholder")
        with patch.dict("os.environ", {}, clear=True), patch(
            "pipeline.model._compat_chat_completion", return_value='{"ok":true}',
        ) as compat:
            content = _sdk_or_compat_completion(
                settings, [{"role": "user", "content": "测试"}],
                max_tokens=12, json_object=True, thinking_mode="disabled",
            )
        self.assertEqual(content, '{"ok":true}')
        self.assertEqual(compat.call_args.kwargs["request_timeout"], settings.request_timeout)

    def test_model_request_has_a_total_wall_clock_timeout(self) -> None:
        settings = ModelSettings(api_key="placeholder", request_timeout=0.01)

        def stalled(*_args: object, **_kwargs: object) -> str:
            time.sleep(0.2)
            return '{"late":true}'

        started = time.monotonic()
        with patch("pipeline.model._sdk_or_compat_completion", side_effect=stalled), self.assertRaisesRegex(
            ModelServiceError, "总墙钟"
        ):
            _bounded_completion(
                settings, [{"role": "user", "content": "测试"}],
                max_tokens=12, json_object=True, thinking_mode="disabled",
            )
        self.assertLess(time.monotonic() - started, 0.1)

    def test_empty_thinking_content_replays_same_json_task_without_thinking(self) -> None:
        settings = ModelSettings(api_key="placeholder", thinking_enabled=True)
        with patch(
            "pipeline.model._sdk_or_compat_completion",
            side_effect=["", '{"ok":true}'],
        ) as completion:
            result = complete_json("系统", "任务", settings, attempts=1, thinking=True)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[0].kwargs["thinking_mode"], "enabled")
        self.assertEqual(completion.call_args_list[1].kwargs["thinking_mode"], "disabled")
        self.assertEqual(
            completion.call_args_list[0].args[1],
            completion.call_args_list[1].args[1],
        )

    def test_thinking_service_failure_replays_without_thinking(self) -> None:
        settings = ModelSettings(api_key="placeholder", thinking_enabled=True)
        with patch(
            "pipeline.model._sdk_or_compat_completion",
            side_effect=[ModelServiceError("thinking timeout"), '{"ok":true}'],
        ) as completion:
            result = complete_json("系统", "任务", settings, attempts=1, thinking=True)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[1].kwargs["thinking_mode"], "disabled")

    def test_balance_failure_does_not_replay_thinking_request(self) -> None:
        settings = ModelSettings(api_key="placeholder", thinking_enabled=True)
        with patch(
            "pipeline.model._sdk_or_compat_completion",
            side_effect=ModelServiceError("模型服务返回 HTTP 402: Insufficient Balance"),
        ) as completion, self.assertRaisesRegex(ModelServiceError, "402"):
            complete_json("系统", "任务", settings, attempts=1, thinking=True)
        self.assertEqual(completion.call_count, 1)

    def test_service_failure_is_not_wrapped_as_invalid_json(self) -> None:
        settings = ModelSettings(api_key="placeholder")
        with patch(
            "pipeline.model._sdk_or_compat_completion",
            side_effect=ModelServiceError("余额不足"),
        ), self.assertRaisesRegex(ModelServiceError, "余额不足"):
            complete_json("系统", "任务", settings, attempts=2)

    def test_invalid_json_is_a_distinct_output_contract_error(self) -> None:
        settings = ModelSettings(api_key="placeholder")
        with patch(
            "pipeline.model._sdk_or_compat_completion",
            return_value="这不是JSON",
        ), self.assertRaises(ModelOutputError):
            complete_json("系统", "任务", settings, attempts=1)


if __name__ == "__main__":
    unittest.main()
