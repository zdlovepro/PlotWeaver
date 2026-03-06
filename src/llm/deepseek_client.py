from typing import Optional, Iterator

from .base import BaseLLMClient, LLMResponse, Message
from .config import DeepSeekConfig


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API client using the OpenAI-compatible SDK."""

    def __init__(self, config: Optional[DeepSeekConfig] = None, api_key: str = "", model: str = "deepseek-chat"):
        if config is None:
            config = DeepSeekConfig(api_key=api_key, model=model)
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai package is required: pip install openai>=1.0.0") from exc
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        client = self._get_client()
        openai_messages = [m.to_dict() for m in messages]
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=openai_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_content = None
        if hasattr(choice.message, "reasoning_content"):
            reasoning_content = choice.message.reasoning_content
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=response.model,
            usage=usage,
        )

    def stream_chat(self, messages: list[Message], **kwargs) -> Iterator[str]:
        client = self._get_client()
        openai_messages = [m.to_dict() for m in messages]
        stream = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=openai_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False
