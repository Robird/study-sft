"""Typed domain objects for sample-projected agentic-context payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


AgenticKind = Literal["belief", "observation", "me"]


@dataclass(frozen=True, slots=True)
class AgenticText:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("AgenticText.text must be a string")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "text",
            "text": self.text,
        }


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
class AgenticAction:
    content: tuple[AgenticText | AgenticOpaquePayload, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.content, (tuple, list)):
            raise ValueError("AgenticAction.content must be a sequence")
        content = tuple(self.content)
        for item in content:
            if not isinstance(item, (AgenticText, AgenticOpaquePayload)):
                raise ValueError("AgenticAction.content items must be AgenticText or AgenticOpaquePayload values")
        object.__setattr__(self, "content", content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "action",
            "content": [item.to_dict() for item in self.content],
        }


@dataclass(frozen=True, slots=True)
class AgenticEntry:
    kind: AgenticKind
    content: tuple[AgenticText | AgenticOpaquePayload | AgenticAction, ...]
    loss: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("AgenticEntry.kind must be a string")
        if not isinstance(self.loss, bool):
            raise ValueError("AgenticEntry.loss must be a boolean")
        if not isinstance(self.content, (tuple, list)):
            raise ValueError("AgenticEntry.content must be a sequence")
        content = tuple(self.content)
        for item in content:
            if not isinstance(item, (AgenticText, AgenticOpaquePayload, AgenticAction)):
                raise ValueError(
                    "AgenticEntry.content items must be AgenticText, AgenticOpaquePayload, or AgenticAction values"
                )
        object.__setattr__(self, "content", content)

    def to_dict(self) -> dict[str, Any]:
        entry = {
            "kind": self.kind,
            "content": [item.to_dict() for item in self.content],
        }
        if self.loss:
            entry["loss"] = True
        return entry


@dataclass(frozen=True, slots=True)
class AgenticContext:
    entries: tuple[AgenticEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, (tuple, list)):
            raise ValueError("AgenticContext.entries must be a sequence")
        entries = tuple(self.entries)
        for entry in entries:
            if not isinstance(entry, AgenticEntry):
                raise ValueError("AgenticContext.entries must contain AgenticEntry values")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
        }
