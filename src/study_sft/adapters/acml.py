"""Adapters from ACML layers into study_sft's current v0 core model."""

from __future__ import annotations

from typing import Any, Literal

from acml import parse_document
from acml.model import Attribute, Document, EntryNode
from acml.semantic_model import (
    SemanticAction,
    SemanticContext,
    SemanticEntry,
    SemanticPayload,
    SemanticText,
    document_to_semantic_context,
)
from study_sft.agentic_context_model import (
    AgenticAction,
    AgenticContext,
    AgenticEntry,
    AgenticOpaquePayload,
    AgenticText,
)


ActionLoweringPolicy = Literal["preserve", "render_text", "reject"]
KindLossPolicy = Literal["none", "all_me", "all_entries"]
ACMLLossPolicy = Literal["explicit"] | KindLossPolicy


def semantic_context_from_acml_document(document: Document) -> SemanticContext:
    return document_to_semantic_context(document)


def semantic_context_from_acml_text(source: str) -> SemanticContext:
    return semantic_context_from_acml_document(parse_document(source))


def agentic_context_from_acml_document(
    document: Document,
    *,
    action_policy: ActionLoweringPolicy = "preserve",
    loss_policy: ACMLLossPolicy = "all_me",
) -> AgenticContext:
    semantic_context = semantic_context_from_acml_document(document)
    entry_losses = _entry_losses_from_document(document, loss_policy=loss_policy)
    return _agentic_context_from_semantic_context(
        semantic_context,
        action_policy=action_policy,
        entry_losses=entry_losses,
    )


def agentic_context_from_acml_text(
    source: str,
    *,
    action_policy: ActionLoweringPolicy = "preserve",
    loss_policy: ACMLLossPolicy = "all_me",
) -> AgenticContext:
    return agentic_context_from_acml_document(
        parse_document(source),
        action_policy=action_policy,
        loss_policy=loss_policy,
    )


def agentic_context_from_acml_record(
    record: dict[str, Any],
    *,
    action_policy: ActionLoweringPolicy = "preserve",
    loss_policy: ACMLLossPolicy = "all_me",
) -> AgenticContext:
    source = record.get("acml")
    if not isinstance(source, str) or not source:
        raise ValueError("acml dataset format expects a non-empty string column named 'acml'")
    return agentic_context_from_acml_text(
        source,
        action_policy=action_policy,
        loss_policy=loss_policy,
    )


def agentic_context_from_semantic_context(
    context: SemanticContext,
    *,
    action_policy: ActionLoweringPolicy = "preserve",
    loss_policy: KindLossPolicy = "all_me",
) -> AgenticContext:
    entry_losses = tuple(_entry_loss_from_kind(entry.kind, loss_policy=loss_policy) for entry in context.entries)
    return _agentic_context_from_semantic_context(
        context,
        action_policy=action_policy,
        entry_losses=entry_losses,
    )


def _agentic_context_from_semantic_context(
    context: SemanticContext,
    *,
    action_policy: ActionLoweringPolicy,
    entry_losses: tuple[bool, ...],
) -> AgenticContext:
    if len(entry_losses) != len(context.entries):
        raise ValueError("entry_losses must align with semantic context entries")
    return AgenticContext(
        entries=tuple(
            _semantic_entry_to_agentic_entry(
                entry,
                loss=entry_losses[index],
                action_policy=action_policy,
            )
            for index, entry in enumerate(context.entries)
        )
    )


def _semantic_entry_to_agentic_entry(
    entry: SemanticEntry,
    *,
    loss: bool,
    action_policy: ActionLoweringPolicy,
) -> AgenticEntry:
    content: list[AgenticText | AgenticOpaquePayload | AgenticAction] = []
    for item in entry.content:
        if isinstance(item, SemanticText):
            content.append(AgenticText(text=item.text))
            continue
        if isinstance(item, SemanticPayload):
            content.append(AgenticOpaquePayload(text=item.text))
            continue
        if isinstance(item, SemanticAction):
            if action_policy == "reject":
                raise ValueError("study_sft v0 adapter cannot lower SemanticAction when action_policy='reject'")
            if action_policy == "render_text":
                content.append(AgenticText(text=_render_semantic_action(item)))
                continue
            content.append(
                AgenticAction(
                    content=tuple(_semantic_action_item_to_agentic_node(child) for child in item.content),
                )
            )
            continue
        raise TypeError(f"unsupported semantic entry content node: {type(item)!r}")
    return AgenticEntry(kind=entry.kind, content=tuple(content), loss=loss)


def _render_semantic_action(action: SemanticAction) -> str:
    pieces: list[str] = []
    for item in action.content:
        if isinstance(item, SemanticText):
            pieces.append(item.text)
        elif isinstance(item, SemanticPayload):
            pieces.append(item.text)
        else:
            raise TypeError(f"unsupported semantic action content node: {type(item)!r}")
    return "".join(pieces)


def _semantic_action_item_to_agentic_node(item: SemanticText | SemanticPayload) -> AgenticText | AgenticOpaquePayload:
    if isinstance(item, SemanticText):
        return AgenticText(text=item.text)
    if isinstance(item, SemanticPayload):
        return AgenticOpaquePayload(text=item.text)
    raise TypeError(f"unsupported semantic action content node: {type(item)!r}")


def _entry_losses_from_document(document: Document, *, loss_policy: ACMLLossPolicy) -> tuple[bool, ...]:
    if loss_policy == "explicit":
        return tuple(_loss_hint_from_entry(entry) for entry in document.entries)
    return tuple(_entry_loss_from_kind(entry.kind, loss_policy=loss_policy) for entry in document.entries)


def _entry_loss_from_kind(kind: str, *, loss_policy: KindLossPolicy) -> bool:
    if loss_policy == "none":
        return False
    if loss_policy == "all_me":
        return kind == "me"
    if loss_policy == "all_entries":
        return True
    raise ValueError(f"unsupported loss_policy: {loss_policy!r}")


def _loss_hint_from_entry(entry: EntryNode) -> bool:
    return _loss_hint_from_attrs(entry.attrs)


def _loss_hint_from_attrs(attrs: tuple[Attribute, ...]) -> bool:
    values = [attr.value for attr in attrs if attr.name == "loss"]
    if not values:
        return False
    if len(values) > 1:
        raise ValueError("ACML entry may not repeat the 'loss' attribute")
    value = values[0]
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("ACML entry 'loss' attribute must be 'true' or 'false'")
