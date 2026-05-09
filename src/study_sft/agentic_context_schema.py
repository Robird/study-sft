"""External schema adapters for the typed agentic-context model."""

from __future__ import annotations

from typing import Any

from study_sft.agentic_context_model import AgenticContext, AgenticMessage, AgenticOpaquePayload


def agentic_context_from_dict(context: dict[str, Any], *, policy) -> AgenticContext:
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    if "version" in context:
        raise ValueError("context.version is not supported in the external schema")
    messages = context.get("messages")
    if not isinstance(messages, list):
        raise ValueError("context must contain a list field named 'messages'")
    return AgenticContext(messages=tuple(_coerce_message(message, policy) for message in messages))


def _coerce_message(message: Any, policy) -> AgenticMessage:
    if not isinstance(message, dict):
        raise ValueError("each message must be an object")
    unsupported_fields = set(message) - {"role", "loss", "content"}
    if unsupported_fields:
        unsupported = ", ".join(sorted(unsupported_fields))
        raise ValueError(f"unsupported message fields: {unsupported}")
    role_value = message.get("role")
    if not isinstance(role_value, str):
        raise ValueError("message.role must be a string")
    role = role_value.strip()
    if role not in policy.allowed_roles:
        raise ValueError(f"unsupported role: {role!r}")
    loss = message.get("loss", False)
    if not isinstance(loss, bool):
        raise ValueError("message.loss must be a boolean")
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("message.content must be a list")
    return AgenticMessage(
        role=role,
        loss=loss,
        content=tuple(_coerce_content_item(item) for item in content),
    )


def _coerce_content_item(item: Any) -> AgenticOpaquePayload:
    if not isinstance(item, dict):
        raise ValueError("message.content items must be objects")
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
