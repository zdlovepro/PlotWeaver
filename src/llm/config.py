from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class DeepSeekConfig(LLMConfig):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"

    @classmethod
    def from_dict(cls, data: dict) -> "DeepSeekConfig":
        return cls(
            model=data.get("model", "deepseek-chat"),
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 4096)),
            timeout=int(data.get("timeout", 120)),
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url", "https://api.deepseek.com"),
        )


@dataclass
class OllamaConfig(LLMConfig):
    provider: str = "ollama"
    model: str = "qwen2.5"
    host: str = "http://localhost:11434"

    @classmethod
    def from_dict(cls, data: dict) -> "OllamaConfig":
        return cls(
            model=data.get("model", "qwen2.5"),
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 4096)),
            timeout=int(data.get("timeout", 120)),
            host=data.get("host", "http://localhost:11434"),
        )
