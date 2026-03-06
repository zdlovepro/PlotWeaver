from typing import Optional, Union

from .base import BaseLLMClient, LLMResponse, Message
from .config import LLMConfig, DeepSeekConfig, OllamaConfig
from .deepseek_client import DeepSeekClient
from .ollama_client import OllamaClient
from .prompt_manager import PromptManager
from .response_parser import extract_json, parse_plot_units, parse_annotation, parse_outline


def create_client(
    provider: str,
    *,
    api_key: str = "",
    model: str = "",
    host: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 120,
    config: Optional[Union[DeepSeekConfig, OllamaConfig]] = None,
) -> BaseLLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: "deepseek" or "ollama"
        api_key: API key (for DeepSeek)
        model: Model name (defaults to provider default)
        host: Ollama host URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        config: Pre-built config object (overrides other args)
    """
    provider = provider.lower()
    if provider == "deepseek":
        if config is None:
            config = DeepSeekConfig(
                api_key=api_key,
                model=model or "deepseek-chat",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        return DeepSeekClient(config=config)
    elif provider == "ollama":
        if config is None:
            config = OllamaConfig(
                host=host,
                model=model or "qwen2.5",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        return OllamaClient(config=config)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'deepseek' or 'ollama'.")


__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "Message",
    "LLMConfig",
    "DeepSeekConfig",
    "OllamaConfig",
    "DeepSeekClient",
    "OllamaClient",
    "PromptManager",
    "extract_json",
    "parse_plot_units",
    "parse_annotation",
    "parse_outline",
    "create_client",
]
