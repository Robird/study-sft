"""Typed domain objects for sample-projected agentic-context payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


AgenticRole = Literal["belief", "observation", "me"]


@dataclass(frozen=True, slots=True)
class AgenticOpaquePayload:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("AgenticOpaquePayload.text must be a string")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "opaque_payload",
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class AgenticMessage:
    role: AgenticRole
    content: tuple[AgenticOpaquePayload, ...]
    loss: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise ValueError("AgenticMessage.role must be a string")
        if not isinstance(self.loss, bool):
            raise ValueError("AgenticMessage.loss must be a boolean")
        if not isinstance(self.content, (tuple, list)):
            raise ValueError("AgenticMessage.content must be a sequence")
        content = tuple(self.content)
        for item in content:
            if not isinstance(item, AgenticOpaquePayload):
                raise ValueError("AgenticMessage.content items must be AgenticOpaquePayload values")
        object.__setattr__(self, "content", content)

    def to_dict(self) -> dict[str, Any]:
        message = {
            "role": self.role,
            "content": [item.to_dict() for item in self.content],
        }
        if self.loss:
            message["loss"] = True
        return message


@dataclass(frozen=True, slots=True)
class AgenticContext:
    messages: tuple[AgenticMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.messages, (tuple, list)):
            raise ValueError("AgenticContext.messages must be a sequence")
        messages = tuple(self.messages)
        for message in messages:
            if not isinstance(message, AgenticMessage):
                raise ValueError("AgenticContext.messages must contain AgenticMessage values")
        object.__setattr__(self, "messages", messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
        }
