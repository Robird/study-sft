"""Shared semantic model for ACML-like authoring inputs.

This layer is intentionally wider than the current study_sft v0 core model but
narrower than the authoring AST/CST. It captures semantic structure while
staying separate from syntax-specific tree details.
"""

from __future__ import annotations

from dataclasses import dataclass

from acml._validation import validated_sequence
from acml.model import ActionNode, Attribute, Document, EntryNode, PayloadNode, TextNode


__all__ = [
    "SemanticText",
    "SemanticPayload",
    "SemanticAction",
    "SemanticEntry",
    "SemanticContext",
    "SemanticDocument",
    "document_to_semantic_context",
    "document_to_semantic_document",
    "semantic_context_to_document",
    "semantic_document_to_document",
]


@dataclass(frozen=True, slots=True)
class SemanticText:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("SemanticText.text must be a string")


@dataclass(frozen=True, slots=True)
class SemanticPayload:
    text: str
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("SemanticPayload.text must be a string")
        attrs = _validated_attrs(self.attrs, field_name="SemanticPayload.attrs")
        object.__setattr__(self, "attrs", attrs)


@dataclass(frozen=True, slots=True)
class SemanticAction:
    content: tuple[SemanticText | SemanticPayload, ...]
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        content = validated_sequence(
            self.content,
            allowed_types=(SemanticText, SemanticPayload),
            field_name="SemanticAction.content",
        )
        attrs = _validated_attrs(self.attrs, field_name="SemanticAction.attrs")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "attrs", attrs)


@dataclass(frozen=True, slots=True)
class SemanticEntry:
    kind: str
    content: tuple[SemanticText | SemanticPayload | SemanticAction, ...]
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("SemanticEntry.kind must be a string")
        content = validated_sequence(
            self.content,
            allowed_types=(SemanticText, SemanticPayload, SemanticAction),
            field_name="SemanticEntry.content",
        )
        attrs = _validated_attrs(self.attrs, field_name="SemanticEntry.attrs")
        if any(attr.name == "kind" for attr in attrs):
            raise ValueError("SemanticEntry.attrs may not repeat promoted attribute 'kind'")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "attrs", attrs)


@dataclass(frozen=True, slots=True)
class SemanticContext:
    """Lossy semantic projection that keeps only the ordered entry stream."""

    entries: tuple[SemanticEntry, ...]

    def __post_init__(self) -> None:
        entries = validated_sequence(
            self.entries,
            allowed_types=(SemanticEntry,),
            field_name="SemanticContext.entries",
        )
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True, slots=True)
class SemanticDocument:
    """Lossless semantic envelope that preserves document-level metadata."""

    version: str
    entries: tuple[SemanticEntry, ...]
    attrs: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.version, str):
            raise ValueError("SemanticDocument.version must be a string")
        entries = validated_sequence(
            self.entries,
            allowed_types=(SemanticEntry,),
            field_name="SemanticDocument.entries",
        )
        attrs = _validated_attrs(self.attrs, field_name="SemanticDocument.attrs")
        if any(attr.name == "version" for attr in attrs):
            raise ValueError("SemanticDocument.attrs may not repeat promoted attribute 'version'")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "attrs", attrs)


def document_to_semantic_document(document: Document) -> SemanticDocument:
    """Project a Document into the lossless semantic envelope."""

    if not isinstance(document, Document):
        raise TypeError("document_to_semantic_document expects a Document")
    return SemanticDocument(
        version=document.version,
        attrs=document.attrs,
        entries=tuple(_entry_to_semantic_entry(entry) for entry in document.entries),
    )


def document_to_semantic_context(document: Document) -> SemanticContext:
    """Project a Document into a lossy entry-only semantic view."""

    semantic_document = document_to_semantic_document(document)
    return SemanticContext(entries=semantic_document.entries)


def semantic_context_to_document(context: SemanticContext, *, version: str = "0") -> Document:
    """Reconstruct a Document from a lossy semantic context using a supplied version."""

    if not isinstance(context, SemanticContext):
        raise TypeError("semantic_context_to_document expects a SemanticContext")
    if not isinstance(version, str):
        raise TypeError("semantic_context_to_document expects version to be a string")
    return semantic_document_to_document(
        SemanticDocument(version=version, entries=context.entries),
    )


def semantic_document_to_document(document: SemanticDocument) -> Document:
    """Project a lossless semantic document back into the ACML syntax model."""

    if not isinstance(document, SemanticDocument):
        raise TypeError("semantic_document_to_document expects a SemanticDocument")
    return Document(
        version=document.version,
        attrs=document.attrs,
        entries=tuple(_semantic_entry_to_entry(entry) for entry in document.entries),
    )


def _entry_to_semantic_entry(entry: EntryNode) -> SemanticEntry:
    return SemanticEntry(
        kind=entry.kind,
        content=tuple(_entry_item_to_semantic_node(item) for item in entry.content),
        attrs=entry.attrs,
    )


def _entry_item_to_semantic_node(
    item: TextNode | PayloadNode | ActionNode,
) -> SemanticText | SemanticPayload | SemanticAction:
    if isinstance(item, TextNode):
        return SemanticText(text=item.text)
    if isinstance(item, PayloadNode):
        return SemanticPayload(text=item.text, attrs=item.attrs)
    if isinstance(item, ActionNode):
        return SemanticAction(
            content=tuple(_action_item_to_semantic_node(child) for child in item.content),
            attrs=item.attrs,
        )
    raise TypeError(f"unsupported ACML entry content node: {type(item)!r}")


def _action_item_to_semantic_node(item: TextNode | PayloadNode) -> SemanticText | SemanticPayload:
    if isinstance(item, TextNode):
        return SemanticText(text=item.text)
    if isinstance(item, PayloadNode):
        return SemanticPayload(text=item.text, attrs=item.attrs)
    raise TypeError(f"unsupported ACML action content node: {type(item)!r}")


def _semantic_entry_to_entry(entry: SemanticEntry) -> EntryNode:
    return EntryNode(
        kind=entry.kind,
        content=tuple(_semantic_node_to_entry_item(item) for item in entry.content),
        attrs=entry.attrs,
    )


def _semantic_node_to_entry_item(
    item: SemanticText | SemanticPayload | SemanticAction,
) -> TextNode | PayloadNode | ActionNode:
    if isinstance(item, SemanticText):
        return TextNode(text=item.text)
    if isinstance(item, SemanticPayload):
        return PayloadNode(text=item.text, attrs=item.attrs)
    if isinstance(item, SemanticAction):
        return ActionNode(
            content=tuple(_semantic_action_node_to_item(child) for child in item.content),
            attrs=item.attrs,
        )
    raise TypeError(f"unsupported semantic entry content node: {type(item)!r}")


def _semantic_action_node_to_item(item: SemanticText | SemanticPayload) -> TextNode | PayloadNode:
    if isinstance(item, SemanticText):
        return TextNode(text=item.text)
    if isinstance(item, SemanticPayload):
        return PayloadNode(text=item.text, attrs=item.attrs)
    raise TypeError(f"unsupported semantic action content node: {type(item)!r}")


def _validated_attrs(
    attrs: tuple[Attribute, ...] | list[Attribute],
    *,
    field_name: str,
) -> tuple[Attribute, ...]:
    return validated_sequence(attrs, allowed_types=(Attribute,), field_name=field_name)
