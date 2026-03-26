from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatTemplate:
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    tool_token: str = "<|tool|>"
    eos_token: str = "</s>"

    def render(self, messages: Iterable[ChatMessage], add_generation_prompt: bool = False) -> str:
        parts: list[str] = []
        role_to_token = {
            "system": self.system_token,
            "user": self.user_token,
            "assistant": self.assistant_token,
            "tool": self.tool_token,
        }
        for message in messages:
            token = role_to_token.get(message.role)
            if token is None:
                raise ValueError(f"Unsupported role {message.role!r}")
            parts.append(f"{token}\n{message.content.strip()}\n{self.eos_token}")
        if add_generation_prompt:
            parts.append(f"{self.assistant_token}\n")
        return "\n".join(parts)


def format_chat(messages: Iterable[dict[str, str]], add_generation_prompt: bool = False) -> str:
    template = ChatTemplate()
    parsed = [ChatMessage(role=item["role"], content=item["content"]) for item in messages]
    return template.render(parsed, add_generation_prompt=add_generation_prompt)

