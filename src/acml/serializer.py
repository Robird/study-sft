"""Serializer for the in-memory ACML model."""

from __future__ import annotations

from acml.model import ActionNode, Attribute, Document, EntryNode, PayloadNode, TextNode


def serialize_document(document: Document) -> str:
    if not isinstance(document, Document):
        raise TypeError("serialize_document expects a Document")
    parts = [_open_tag("acml", [Attribute("version", document.version), *document.attrs])]
    for entry in document.entries:
        parts.append(_serialize_entry(entry))
    parts.append("</acml>")
    return "".join(parts)


def _serialize_entry(entry: EntryNode) -> str:
    attrs = [Attribute("kind", entry.kind), *entry.attrs]
    parts = [_open_tag("acml:entry", attrs)]
    for item in entry.content:
        parts.append(_serialize_entry_item(item))
    parts.append("</acml:entry>")
    return "".join(parts)


def _serialize_entry_item(item: TextNode | PayloadNode | ActionNode) -> str:
    if isinstance(item, TextNode):
        return _escape_text(item.text)
    if isinstance(item, PayloadNode):
        return _serialize_payload(item)
    if isinstance(item, ActionNode):
        return _serialize_action(item)
    raise TypeError(f"unsupported entry content node: {type(item)!r}")


def _serialize_payload(payload: PayloadNode) -> str:
    return f"{_open_tag('acml:payload', payload.attrs)}{_escape_text(payload.text)}</acml:payload>"


def _serialize_action(action: ActionNode) -> str:
    parts = [_open_tag("acml:action", action.attrs)]
    for item in action.content:
        if isinstance(item, TextNode):
            parts.append(_escape_text(item.text))
        elif isinstance(item, PayloadNode):
            parts.append(_serialize_payload(item))
        else:
            raise TypeError(f"unsupported action content node: {type(item)!r}")
    parts.append("</acml:action>")
    return "".join(parts)


def _open_tag(name: str, attrs: list[Attribute] | tuple[Attribute, ...]) -> str:
    serialized_attrs = "".join(f' {attr.name}="{_serialize_attribute_value(attr.value)}"' for attr in attrs)
    return f"<{name}{serialized_attrs}>"


def _serialize_attribute_value(value: str) -> str:
    if any(char in value for char in ('"', "\n", "\r", "<", ">")):
        raise ValueError("ACML attribute values may not contain double quotes, angle brackets, or newlines")
    return value


def _escape_text(text: str) -> str:
    return text.replace("</acml", "&lt;/acml").replace("<acml", "&lt;acml")
