"""In-memory ACML model.

The model is intentionally small and preserves enough structure for round-trip
serialization while remaining separate from any project-specific typed IR.
"""

from __future__ import annotations

from dataclasses import dataclass

from acml._validation import validated_sequence


__all__ = [
    "Attribute",
    "TextNode",
    "PayloadNode",
    "ActionNode",
    "EntryNode",
    "Document",
]


@dataclass(frozen=True, slots=True)
class Attribute:
    name: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("Attribute.name must be a string")
        if not isinstance(self.value, str):
            raise ValueError("Attribute.value must be a string")


@dataclass(frozen=True, slots=True)
class TextNode:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("TextNode.text must be a string")


@dataclass(frozen=True, slots=True)
class PayloadNode:
    text: str
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("PayloadNode.text must be a string")
        attrs = _validated_attrs(self.attrs, field_name="PayloadNode.attrs")
        object.__setattr__(self, "attrs", attrs)


@dataclass(frozen=True, slots=True)
class ActionNode:
    content: tuple[ActionContentNode, ...]
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        content = validated_sequence(
            self.content,
            allowed_types=(TextNode, PayloadNode),
            field_name="ActionNode.content",
        )
        attrs = _validated_attrs(self.attrs, field_name="ActionNode.attrs")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "attrs", attrs)


@dataclass(frozen=True, slots=True)
class EntryNode:
    kind: str
    content: tuple[EntryContentNode, ...]
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("EntryNode.kind must be a string")
        content = validated_sequence(
            self.content,
            allowed_types=(TextNode, PayloadNode, ActionNode),
            field_name="EntryNode.content",
        )
        attrs = _validated_attrs(self.attrs, field_name="EntryNode.attrs")
        if any(attr.name == "kind" for attr in attrs):
            raise ValueError("EntryNode.attrs may not repeat promoted attribute 'kind'")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "attrs", attrs)


@dataclass(frozen=True, slots=True)
class Document:
    version: str
    entries: tuple[EntryNode, ...]
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.version, str):
            raise ValueError("Document.version must be a string")
        entries = validated_sequence(
            self.entries,
            allowed_types=(EntryNode,),
            field_name="Document.entries",
        )
        attrs = _validated_attrs(self.attrs, field_name="Document.attrs")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "attrs", attrs)


ActionContentNode = TextNode | PayloadNode
EntryContentNode = TextNode | PayloadNode | ActionNode


def _validated_attrs(
    attrs: tuple[Attribute, ...] | list[Attribute],
    *,
    field_name: str,
) -> tuple[Attribute, ...]:
    return validated_sequence(attrs, allowed_types=(Attribute,), field_name=field_name)
