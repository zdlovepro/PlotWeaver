from typing import Optional, Iterator

from .base import BaseLLMClient, LLMResponse, Message
from .config import OllamaConfig


class OllamaClient(BaseLLMClient):
    """Ollama local model client using httpx direct HTTP calls."""

    def __init__(self, config: Optional[OllamaConfig] = None, host: str = "http://localhost:11434", model: str = "qwen2.5"):
        if config is None:
            config = OllamaConfig(host=host, model=model)
        self.config = config

    def _get_httpx(self):
        try:
            import httpx
            return httpx
        except ImportError as exc:
            raise ImportError("httpx package is required: pip install httpx>=0.25.0") from exc

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        httpx = self._get_httpx()
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(f"{self.config.host}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        content = data.get("message", {}).get("content", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model),
            usage=usage,
        )

    def stream_chat(self, messages: list[Message], **kwargs) -> Iterator[str]:
        httpx = self._get_httpx()
        import json as _json
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            with client.stream("POST", f"{self.config.host}/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = _json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                            if data.get("done"):
                                break
                        except _json.JSONDecodeError:
                            continue

    def is_available(self) -> bool:
        try:
            httpx = self._get_httpx()
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.config.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return a list of locally available Ollama model names."""
        httpx = self._get_httpx()
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{self.config.host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def get_model_info(self, model_name: str) -> dict:
        """Return detailed info for the given model."""
        httpx = self._get_httpx()
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{self.config.host}/api/show", json={"name": model_name})
            resp.raise_for_status()
            return resp.json()
