"""External schema adapters for the typed agentic-context model."""

from __future__ import annotations

from typing import Any

from study_sft.agentic_context_model import AgenticContext, AgenticEntry, AgenticOpaquePayload


def agentic_context_from_dict(context: dict[str, Any], *, policy) -> AgenticContext:
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    if "version" in context:
        raise ValueError("context.version is not supported in the external schema")
    entries = context.get("entries")
    if not isinstance(entries, list):
        raise ValueError("context must contain a list field named 'entries'")
    return AgenticContext(entries=tuple(_coerce_entry(entry, policy) for entry in entries))


def _coerce_entry(entry: Any, policy) -> AgenticEntry:
    if not isinstance(entry, dict):
        raise ValueError("each entry must be an object")
    unsupported_fields = set(entry) - {"kind", "loss", "content"}
    if unsupported_fields:
        unsupported = ", ".join(sorted(unsupported_fields))
        raise ValueError(f"unsupported entry fields: {unsupported}")
    kind_value = entry.get("kind")
    if not isinstance(kind_value, str):
        raise ValueError("entry.kind must be a string")
    kind = kind_value.strip()
    if kind not in policy.allowed_kinds:
        raise ValueError(f"unsupported kind: {kind!r}")
    loss = entry.get("loss", False)
    if not isinstance(loss, bool):
        raise ValueError("entry.loss must be a boolean")
    content = entry.get("content")
    if not isinstance(content, list):
        raise ValueError("entry.content must be a list")
    return AgenticEntry(
        kind=kind,
        loss=loss,
        content=tuple(_coerce_content_item(item) for item in content),
    )


def _coerce_content_item(item: Any) -> AgenticOpaquePayload:
    if not isinstance(item, dict):
        raise ValueError("entry.content items must be objects")
    kind = item.get("kind")
    if kind != "opaque_payload":
        raise ValueError(f"unsupported content kind: {kind!r}")
    unsupported_fields = set(item) - {"kind", "text"}
    if unsupported_fields:
        unsupported = ", ".join(sorted(unsupported_fields))
        raise ValueError(f"unsupported opaque_payload fields: {unsupported}")
    text = item.get("text")
    if not isinstance(text, str):
        raise ValueError("opaque_payload.text must be a string")
    return AgenticOpaquePayload(text=text)
