from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    content: str
    reasoning_content: Optional[str] = None
    model: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content


from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send a list of messages and return a response."""

    @abstractmethod
    def stream_chat(self, messages: list[Message], **kwargs):
        """Send messages and stream the response as a generator of str chunks."""

    def ask(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Convenience method: send a single user prompt, return text."""
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        response = self.chat(messages, **kwargs)
        return response.content

    def ask_json(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Convenience method: ask for JSON output, return raw response text."""
        return self.ask(prompt, system=system, **kwargs)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is reachable."""
