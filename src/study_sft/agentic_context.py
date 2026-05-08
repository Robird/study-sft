"""Token-id safe serialization for experimental Agentic context data.

This module keeps the experimental external schema intentionally small:

- messages use ``content`` instead of block wrappers
- content items are only inline strings / ``opaque_payload`` / ``structured_region``
- canonical ``text`` nodes are internal output from normalization
- opaque untrusted payloads are represented by ``opaque_payload``
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


__all__ = [
    "AgenticContextEncoder",
    "AgenticContextPolicy",
    "AgenticTokenTable",
    "DEFAULT_AGENTIC_CONTEXT_POLICY",
    "QWEN3_AGENTIC_TOKEN_TABLE",
    "DebugEncodedContext",
    "EncodedContext",
    "EncodedText",
    "Span",
    "mark_training_targets",
]


class _TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


ENCODING_VERSION = "agentic-context-v0"

STRUCTURE_MESSAGE_START = "message_start"
STRUCTURE_MESSAGE_END = "message_end"
STRUCTURE_OPAQUE_PAYLOAD_START = "opaque_payload_start"
STRUCTURE_OPAQUE_PAYLOAD_END = "opaque_payload_end"
STRUCTURE_STRUCTURED_REGION_START = "structured_region_start"
STRUCTURE_STRUCTURED_REGION_END = "structured_region_end"

SPAN_KIND_STRUCTURE = "structure"
SPAN_KIND_ROLE = "role"
SPAN_KIND_NEWLINE = "newline"
SPAN_KIND_TEXT = "text"
SPAN_KIND_OPAQUE_PAYLOAD = "opaque_payload"

NODE_KIND_TEXT = "text"
NODE_KIND_OPAQUE_PAYLOAD = "opaque_payload"
NODE_KIND_STRUCTURED_REGION = "structured_region"


@dataclass(frozen=True, slots=True)
class AgenticTokenTable:
    """Semantic structure-token names mapped to the current borrowed token texts."""

    message_start: int = 151644
    message_end: int = 151645
    opaque_payload_start: int = 151648
    opaque_payload_end: int = 151649
    structured_region_start: int = 151650
    structured_region_end: int = 151651

    message_start_text: str = "<|im_start|>"
    message_end_text: str = "<|im_end|>"
    opaque_payload_start_text: str = "<|box_start|>"
    opaque_payload_end_text: str = "<|box_end|>"
    structured_region_start_text: str = "<|quad_start|>"
    structured_region_end_text: str = "<|quad_end|>"

    _id_by_name: dict[str, int] = field(init=False, repr=False, compare=False)
    _text_by_name: dict[str, str] = field(init=False, repr=False, compare=False)
    _reserved_ids: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        id_by_name = {
            STRUCTURE_MESSAGE_START: self.message_start,
            STRUCTURE_MESSAGE_END: self.message_end,
            STRUCTURE_OPAQUE_PAYLOAD_START: self.opaque_payload_start,
            STRUCTURE_OPAQUE_PAYLOAD_END: self.opaque_payload_end,
            STRUCTURE_STRUCTURED_REGION_START: self.structured_region_start,
            STRUCTURE_STRUCTURED_REGION_END: self.structured_region_end,
        }
        text_by_name = {
            STRUCTURE_MESSAGE_START: self.message_start_text,
            STRUCTURE_MESSAGE_END: self.message_end_text,
            STRUCTURE_OPAQUE_PAYLOAD_START: self.opaque_payload_start_text,
            STRUCTURE_OPAQUE_PAYLOAD_END: self.opaque_payload_end_text,
            STRUCTURE_STRUCTURED_REGION_START: self.structured_region_start_text,
            STRUCTURE_STRUCTURED_REGION_END: self.structured_region_end_text,
        }
        object.__setattr__(self, "_id_by_name", id_by_name)
        object.__setattr__(self, "_text_by_name", text_by_name)
        object.__setattr__(self, "_reserved_ids", frozenset(id_by_name.values()))

    def id_by_name(self) -> dict[str, int]:
        return dict(self._id_by_name)

    def text_by_name(self) -> dict[str, str]:
        return dict(self._text_by_name)

    def reserved_ids(self) -> frozenset[int]:
        return self._reserved_ids


QWEN3_AGENTIC_TOKEN_TABLE = AgenticTokenTable()


@dataclass(frozen=True, slots=True)
class AgenticContextPolicy:
    token_table: AgenticTokenTable = QWEN3_AGENTIC_TOKEN_TABLE
    allowed_roles: tuple[str, ...] = (
        "observation",
        "belief",
        "me",
        "system",
        "user",
        "assistant",
        "tool",
    )
    extra_reserved_ids: tuple[int, ...] = ()
    max_depth: int = 8

    _reserved_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    _allowed_roles: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_reserved_ids",
            frozenset((*self.token_table.reserved_ids(), *self.extra_reserved_ids)),
        )
        object.__setattr__(self, "_allowed_roles", frozenset(self.allowed_roles))

    def reserved_ids(self) -> frozenset[int]:
        return self._reserved_ids


DEFAULT_AGENTIC_CONTEXT_POLICY = AgenticContextPolicy()


@dataclass(frozen=True, slots=True)
class EncodedText:
    input_ids: list[int]
    encoding: str
    text: str


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    role: str
    kind: str


@dataclass(frozen=True, slots=True)
class EncodedContext:
    input_ids: list[int]
    loss_mask: list[int]
    encoding_version: str = ENCODING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding_version": self.encoding_version,
            "input_ids": self.input_ids,
            "loss_mask": self.loss_mask,
        }


@dataclass(frozen=True, slots=True)
class DebugEncodedContext:
    encoded: EncodedContext
    spans: tuple[Span, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded": self.encoded.to_dict(),
            "spans": [asdict(span) for span in self.spans],
        }


@dataclass(frozen=True, slots=True)
class _TextNode:
    text: str


@dataclass(frozen=True, slots=True)
class _OpaquePayloadNode:
    text: str


@dataclass(frozen=True, slots=True)
class _StructuredRegionNode:
    items: tuple["_ContentNode", ...]


_ContentNode = _TextNode | _OpaquePayloadNode | _StructuredRegionNode


@dataclass(frozen=True, slots=True)
class _NormalizedMessage:
    role: str
    loss: bool
    content: tuple[_ContentNode, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedContext:
    messages: tuple[_NormalizedMessage, ...]

_ROLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class _ContextLayout:
    structure_ids: dict[str, int]
    id_to_structure_name: dict[int, str]
    role_prefix_ids: dict[str, list[int]]
    ordered_role_prefixes: tuple[tuple[str, list[int]], ...]
    newline_ids: list[int]


def _encode_opaque_payload_text(
    text: Any,
    tokenizer: _TokenizerLike,
    reserved_ids: frozenset[int],
) -> EncodedText:
    """Encode payload text without allowing reserved structure token ids."""

    raw_text = "" if text is None else str(text)
    escaped_text = _escape_opaque_payload_text(raw_text)
    escaped_ids = tokenizer.encode(escaped_text, add_special_tokens=False)
    if _contains_reserved_id(escaped_ids, reserved_ids):
        raise ValueError("escaped untrusted text still produced reserved token ids")
    return EncodedText(input_ids=escaped_ids, encoding="text-escaped", text=escaped_text)


def _escape_opaque_payload_text(text: str) -> str:
    """Return a reversible text representation that breaks control-token text."""

    replacements = {
        "\\": "\\\\",
        "<": "\\u003c",
        ">": "\\u003e",
    }
    return "".join(replacements.get(char, char) for char in text)


def _unescape_opaque_payload_text(text: str) -> str:
    """Reverse ``_escape_opaque_payload_text`` for debugging and round-trip tests."""

    output: list[str] = []
    index = 0
    escapes = {
        "\\\\": "\\",
        "\\u003c": "<",
        "\\u003e": ">",
    }
    while index < len(text):
        for escaped, raw in escapes.items():
            if text.startswith(escaped, index):
                output.append(raw)
                index += len(escaped)
                break
        else:
            output.append(text[index])
            index += 1
    return "".join(output)


def _parse_context_json(
    context: dict[str, Any],
    policy: AgenticContextPolicy = DEFAULT_AGENTIC_CONTEXT_POLICY,
) -> _NormalizedContext:
    """Validate external JSON and return the normalized typed IR."""

    if not isinstance(context, dict):
        raise ValueError("context must be an object")

    if "version" in context:
        raise ValueError("context.version is not supported in the external schema")

    messages = context.get("messages")
    if not isinstance(messages, list):
        raise ValueError("context must contain a list field named 'messages'")

    return _NormalizedContext(messages=tuple(_normalize_message(message, policy) for message in messages))


def mark_training_targets(
    context: dict[str, Any],
    target_message_indexes: int | Iterable[int] = -1,
) -> dict[str, Any]:
    """Return a shallow context copy with loss enabled only for selected messages."""

    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    messages = context.get("messages")
    if not isinstance(messages, list):
        raise ValueError("context must contain a list field named 'messages'")

    if not messages:
        target_indexes = _resolve_target_message_indexes(target_message_indexes, 0)
        if target_indexes:
            raise ValueError("empty messages cannot have training targets")
        marked_context = dict(context)
        marked_context["messages"] = []
        return marked_context

    target_indexes = _resolve_target_message_indexes(target_message_indexes, len(messages))
    marked_context = dict(context)
    marked_context["messages"] = _copy_messages_with_loss(messages, target_indexes)
    return marked_context


def _copy_messages_with_loss(messages: list[Any], target_indexes: frozenset[int]) -> list[dict[str, Any]]:
    marked_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        copied_message = dict(message)
        if index in target_indexes:
            copied_message["loss"] = True
        else:
            copied_message.pop("loss", None)
        marked_messages.append(copied_message)
    return marked_messages


class AgenticContextEncoder:
    """Reusable Agentic context encoder for an immutable tokenizer/policy pair."""

    def __init__(
        self,
        tokenizer: _TokenizerLike,
        policy: AgenticContextPolicy = DEFAULT_AGENTIC_CONTEXT_POLICY,
    ) -> None:
        _ensure_tokenizer_matches_policy(tokenizer, policy)
        self.tokenizer = tokenizer
        self.policy = policy
        self.reserved_ids = policy.reserved_ids()
        self._layout: _ContextLayout | None = None

    def encode_payload(self, text: Any) -> EncodedText:
        """Encode opaque payload text without allowing reserved structure ids."""

        return _encode_opaque_payload_text(text, self.tokenizer, self.reserved_ids)

    def encode_context(
        self,
        context: dict[str, Any],
        *,
        validate: bool = False,
    ) -> EncodedContext:
        """Serialize an Agentic context JSON object to token ids and loss mask."""

        normalized = _parse_context_json(context, self.policy)
        encoded = self._encode_normalized_context(normalized, collect_debug=False)
        if validate:
            self.validate(encoded)
        return encoded

    def encode_context_with_debug(
        self,
        context: dict[str, Any],
        *,
        validate: bool = False,
    ) -> DebugEncodedContext:
        """Serialize an Agentic context JSON object to encoded output plus debug trace."""

        normalized = _parse_context_json(context, self.policy)
        debug_encoded = self._encode_normalized_context(normalized, collect_debug=True)
        if validate:
            self.validate_debug(debug_encoded)
        return debug_encoded

    def _encode_normalized_context(
        self,
        normalized: _NormalizedContext,
        *,
        collect_debug: bool,
    ) -> EncodedContext | DebugEncodedContext:
        self._layout_or_build()
        builder = _ContextBuilder(encoder=self, collect_debug=collect_debug)
        for message in normalized.messages:
            builder.serialize_message(message)
        if collect_debug:
            return builder.debug_encoded()
        return builder.encoded()

    def _layout_or_build(self) -> _ContextLayout:
        if self._layout is None:
            self._layout = self._build_layout()
        return self._layout

    def _build_layout(self) -> _ContextLayout:
        structure_ids = self.policy.token_table.id_by_name()
        role_prefix_ids = {
            role: self._encode_checked_inline(f"{role}\n", kind=SPAN_KIND_ROLE)
            for role in self.policy.allowed_roles
        }
        newline_ids = self._encode_checked_inline("\n", kind=SPAN_KIND_NEWLINE)
        return _ContextLayout(
            structure_ids=structure_ids,
            id_to_structure_name={token_id: name for name, token_id in structure_ids.items()},
            role_prefix_ids=role_prefix_ids,
            ordered_role_prefixes=tuple(
                sorted(
                    ((role, list(ids)) for role, ids in role_prefix_ids.items()),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
            ),
            newline_ids=newline_ids,
        )

    def _encode_checked_inline(self, text: str, *, kind: str) -> list[int]:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if _contains_reserved_id(input_ids, self.reserved_ids):
            raise ValueError(f"checked-inline text for kind={kind!r} produced reserved token ids")
        return input_ids

    def validate(self, encoded: EncodedContext) -> None:
        layout = self._layout_or_build()
        _walk_encoded_spans(encoded, self.policy, layout)

    def validate_debug(self, debug_encoded: DebugEncodedContext) -> None:
        layout = self._layout_or_build()
        encoded_spans = _walk_encoded_spans(debug_encoded.encoded, self.policy, layout)
        debug_spans = _validate_debug_context_spans(debug_encoded, self.policy, layout)
        if debug_spans != encoded_spans:
            raise ValueError("debug spans do not match the encoded token trace")


class _ContextBuilder:
    def __init__(self, encoder: AgenticContextEncoder, *, collect_debug: bool) -> None:
        self.encoder = encoder
        self.policy = encoder.policy
        self.layout = encoder._layout_or_build()
        self.collect_debug = collect_debug
        self.input_ids: list[int] = []
        self.spans: list[Span] = []
        self.loss_mask: list[int] = []

    def encoded(self) -> EncodedContext:
        return EncodedContext(
            input_ids=self.input_ids,
            loss_mask=self.loss_mask,
            encoding_version=ENCODING_VERSION,
        )

    def debug_encoded(self) -> DebugEncodedContext:
        if not self.collect_debug:
            raise RuntimeError("debug trace collection was not enabled")
        encoded = self.encoded()
        return DebugEncodedContext(encoded=encoded, spans=_canonicalize_debug_spans(tuple(self.spans)))

    def serialize_message(self, message: _NormalizedMessage) -> None:
        role = message.role
        message_loss = message.loss

        self._append_structure(STRUCTURE_MESSAGE_START, role=role, loss=message_loss)
        self._append_ids(
            self.layout.role_prefix_ids[role],
            role=role,
            loss=message_loss,
            kind=SPAN_KIND_ROLE,
        )

        for node in message.content:
            self.serialize_node(node, role=role, loss=message_loss, structure_depth=0)

        self._append_structure(STRUCTURE_MESSAGE_END, role=role, loss=message_loss)
        self._append_ids(
            self.layout.newline_ids,
            role=role,
            loss=message_loss,
            kind=SPAN_KIND_NEWLINE,
        )

    def serialize_node(
        self,
        node: _ContentNode,
        role: str,
        loss: bool,
        structure_depth: int,
    ) -> None:
        if isinstance(node, _TextNode):
            self._append_checked_text(node.text, role=role, loss=loss, kind=SPAN_KIND_TEXT)
            return

        if isinstance(node, _OpaquePayloadNode):
            self._serialize_opaque_payload(node, role=role, loss=loss, structure_depth=structure_depth)
            return

        if isinstance(node, _StructuredRegionNode):
            self._serialize_structured_region(node, role=role, loss=loss, structure_depth=structure_depth)
            return

        raise TypeError(f"unknown normalized node type: {type(node)!r}")

    def _serialize_opaque_payload(
        self,
        node: _OpaquePayloadNode,
        *,
        role: str,
        loss: bool,
        structure_depth: int,
    ) -> None:
        if structure_depth + 1 > self.policy.max_depth:
            raise ValueError(f"max structure depth exceeded: {self.policy.max_depth}")
        self._append_structure(STRUCTURE_OPAQUE_PAYLOAD_START, role=role, loss=loss)
        encoded = self.encoder.encode_payload(node.text)
        self._append_encoded_text(encoded, role=role, loss=loss, kind=SPAN_KIND_OPAQUE_PAYLOAD)
        self._append_structure(STRUCTURE_OPAQUE_PAYLOAD_END, role=role, loss=loss)

    def _serialize_structured_region(
        self,
        node: _StructuredRegionNode,
        *,
        role: str,
        loss: bool,
        structure_depth: int,
    ) -> None:
        if structure_depth + 1 > self.policy.max_depth:
            raise ValueError(f"max structure depth exceeded: {self.policy.max_depth}")
        self._append_structure(STRUCTURE_STRUCTURED_REGION_START, role=role, loss=loss)
        for item in node.items:
            self.serialize_node(item, role, loss, structure_depth + 1)
        self._append_structure(STRUCTURE_STRUCTURED_REGION_END, role=role, loss=loss)

    def _append_structure(self, name: str, role: str, loss: bool) -> None:
        token_id = self.layout.structure_ids[name]
        start = len(self.input_ids)
        self.input_ids.append(token_id)
        self.loss_mask.append(1 if loss else 0)
        if self.collect_debug:
            self.spans.append(Span(start=start, end=start + 1, role=role, kind=SPAN_KIND_STRUCTURE))

    def _append_checked_text(self, text: str, *, role: str, loss: bool, kind: str) -> None:
        self._append_ids(self.encoder._encode_checked_inline(text, kind=kind), role=role, loss=loss, kind=kind)

    def _append_encoded_text(self, encoded: EncodedText, *, role: str, loss: bool, kind: str) -> None:
        self._append_ids(encoded.input_ids, role=role, loss=loss, kind=kind)

    def _append_ids(self, input_ids: list[int], *, role: str, loss: bool, kind: str) -> None:
        if not input_ids:
            return
        start = len(self.input_ids)
        self.input_ids.extend(input_ids)
        self.loss_mask.extend([1 if loss else 0] * len(input_ids))
        if self.collect_debug:
            self.spans.append(Span(start=start, end=len(self.input_ids), role=role, kind=kind))


def _normalize_message(message: Any, policy: AgenticContextPolicy) -> _NormalizedMessage:
    if not isinstance(message, dict):
        raise ValueError("each message must be an object")
    if "blocks" in message:
        raise ValueError("message.blocks is not supported; use message.content")
    if "source" in message:
        raise ValueError("message.source is not supported in the v0 schema")
    unsupported_fields = set(message) - {"role", "loss", "content"}
    if unsupported_fields:
        unsupported = ", ".join(sorted(unsupported_fields))
        raise ValueError(f"unsupported message fields: {unsupported}")

    role = _validate_role(message.get("role"), policy)
    message_loss = _resolve_loss(message.get("loss"), default=False)
    if "content" not in message:
        raise ValueError("message.content must be a list")
    content = _normalize_content(message.get("content"), policy=policy)
    return _NormalizedMessage(role=role, loss=message_loss, content=content)


def _normalize_content(value: Any, policy: AgenticContextPolicy) -> tuple[_ContentNode, ...]:
    if not isinstance(value, list):
        raise ValueError("message.content must be a list")
    nodes: list[_ContentNode] = []
    for item in value:
        nodes.append(_normalize_node(item, policy=policy))
    return tuple(nodes)


def _normalize_node(node: Any, policy: AgenticContextPolicy) -> _ContentNode:
    if isinstance(node, str):
        return _TextNode(text=node)
    if not isinstance(node, dict):
        raise ValueError("content node must be a string or object")
    if "trust" in node:
        raise ValueError("trust is not supported in the v0 schema")
    if "content" in node:
        raise ValueError("node.content is not supported; use text or items")
    if "provenance" in node:
        raise ValueError("node.provenance is not supported in the v0 schema")

    kind = _required_str(node.get("kind"), field_name="content node kind").strip()
    if not kind:
        raise ValueError("content node kind must not be empty")

    if kind == NODE_KIND_TEXT:
        raise ValueError("external text nodes are not supported; use a string or opaque_payload")
    if "encoding_mode" in node:
        raise ValueError("external encoding_mode is not supported")

    if kind == NODE_KIND_OPAQUE_PAYLOAD:
        unsupported_fields = set(node) - {"kind", "text"}
        if unsupported_fields:
            unsupported = ", ".join(sorted(unsupported_fields))
            raise ValueError(f"unsupported opaque_payload fields: {unsupported}")
        return _OpaquePayloadNode(text=_required_str(node.get("text"), field_name="opaque_payload.text"))

    if kind == NODE_KIND_STRUCTURED_REGION:
        unsupported_fields = set(node) - {"kind", "items"}
        if unsupported_fields:
            unsupported = ", ".join(sorted(unsupported_fields))
            raise ValueError(f"unsupported structured_region fields: {unsupported}")
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{kind}.items must be a list")
        normalized_items: list[_ContentNode] = []
        for item in items:
            normalized_items.append(_normalize_node(item, policy=policy))
        return _StructuredRegionNode(items=tuple(normalized_items))

    raise ValueError(f"unsupported node kind: {kind!r}")


def _validate_role(role_value: Any, policy: AgenticContextPolicy) -> str:
    role = _required_str(role_value, field_name="message.role").strip()
    if not _ROLE_RE.fullmatch(role):
        raise ValueError(f"invalid role name: {role!r}")
    if role not in policy._allowed_roles:
        raise ValueError(f"unsupported role: {role!r}")
    return role


def _resolve_loss(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("message.loss must be a boolean")


def _resolve_target_message_indexes(indexes: int | Iterable[int], message_count: int) -> frozenset[int]:
    if isinstance(indexes, bool):
        raise ValueError("target_message_indexes must be an int or iterable of ints")
    if isinstance(indexes, int):
        raw_indexes = [indexes]
    elif isinstance(indexes, Iterable) and not isinstance(indexes, (str, bytes)):
        raw_indexes = list(indexes)
    else:
        raise ValueError("target_message_indexes must be an int or iterable of ints")

    if message_count == 0:
        if raw_indexes in ([], [-1]):
            return frozenset()
        raise ValueError(f"target message index out of range: {raw_indexes[0]}")

    resolved_indexes: set[int] = set()
    for index in raw_indexes:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("target_message_indexes must contain only ints")
        resolved_index = index + message_count if index < 0 else index
        if resolved_index < 0 or resolved_index >= message_count:
            raise ValueError(f"target message index out of range: {index}")
        resolved_indexes.add(resolved_index)
    return frozenset(resolved_indexes)


def _required_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _contains_reserved_id(input_ids: list[int], reserved_ids: frozenset[int]) -> bool:
    return any(token_id in reserved_ids for token_id in input_ids)


def _validate_encoding_version(encoding_version: str) -> None:
    if encoding_version != ENCODING_VERSION:
        raise ValueError(
            f"unsupported encoding_version: {encoding_version!r}; expected {ENCODING_VERSION!r}"
        )


@dataclass(slots=True)
class _GrammarMachine:
    policy: AgenticContextPolicy
    current_role: str | None = None
    current_loss: int | None = None
    expecting_role: bool = False
    expecting_newline: bool = False
    structure_stack: list[str] = field(default_factory=list)

    def on_message_start(self, role: str, loss: int) -> None:
        if self.expecting_newline:
            raise ValueError("message_end is not followed by a trailing newline")
        if self.current_role is not None or self.expecting_role or self.structure_stack:
            raise ValueError("message_start must begin a new top-level message")
        self.current_role = role
        self.current_loss = loss
        self.expecting_role = True

    def on_role(self, role: str, loss: int) -> None:
        if self.current_role is None:
            raise ValueError("role appears outside message framing")
        if role != self.current_role:
            raise ValueError("all spans in a message must share the same role")
        if self.current_loss != loss:
            raise ValueError("loss_mask must be constant within message role prefix")
        if not self.expecting_role:
            raise ValueError("role span is only allowed immediately after message_start")
        self.expecting_role = False

    def on_text(self, kind: str, role: str, loss: int) -> None:
        if self.expecting_newline:
            raise ValueError("expected trailing newline after message_end")
        if self.current_role is None:
            raise ValueError("non-structure span appears outside message framing")
        if role != self.current_role:
            raise ValueError("all spans in a message must share the same role")
        if self.current_loss != loss:
            raise ValueError("loss_mask must be constant within a message")
        if self.expecting_role:
            raise ValueError("message_start is not followed by a role span")

        inside_opaque_payload = bool(self.structure_stack) and self.structure_stack[-1] == STRUCTURE_OPAQUE_PAYLOAD_START
        if inside_opaque_payload and kind != SPAN_KIND_OPAQUE_PAYLOAD:
            raise ValueError("opaque payload may only contain opaque_payload spans")
        if kind == SPAN_KIND_OPAQUE_PAYLOAD and not inside_opaque_payload:
            raise ValueError("opaque_payload span must be enclosed by opaque payload structure spans")

    def on_structure(self, name: str, role: str, loss: int) -> None:
        if self.expecting_newline:
            raise ValueError("expected trailing newline after message_end")
        if name == STRUCTURE_MESSAGE_START:
            self.on_message_start(role, loss)
            return

        if self.current_role is None:
            raise ValueError("structure span appears outside message framing")
        if role != self.current_role:
            raise ValueError("all spans in a message must share the same role")
        if self.current_loss != loss:
            raise ValueError("loss_mask must be constant within a message")
        if self.expecting_role:
            raise ValueError("message_start must be followed by a role span")

        if name == STRUCTURE_MESSAGE_END:
            if self.structure_stack:
                raise ValueError(f"message_end cannot close while nested structure is open: {self.structure_stack[-1]}")
            self.expecting_newline = True
            return
        if name in {STRUCTURE_OPAQUE_PAYLOAD_START, STRUCTURE_STRUCTURED_REGION_START}:
            if self.structure_stack and self.structure_stack[-1] == STRUCTURE_OPAQUE_PAYLOAD_START:
                raise ValueError("opaque payload is opaque and cannot contain nested structure")
            self.structure_stack.append(name)
            if len(self.structure_stack) > self.policy.max_depth:
                raise ValueError(f"max structure depth exceeded: {self.policy.max_depth}")
            return
        if name == STRUCTURE_OPAQUE_PAYLOAD_END:
            if not self.structure_stack or self.structure_stack[-1] != STRUCTURE_OPAQUE_PAYLOAD_START:
                raise ValueError("opaque_payload_end does not match current structure stack")
            self.structure_stack.pop()
            return
        if name == STRUCTURE_STRUCTURED_REGION_END:
            if not self.structure_stack or self.structure_stack[-1] != STRUCTURE_STRUCTURED_REGION_START:
                raise ValueError("structured_region_end does not match current structure stack")
            self.structure_stack.pop()
            return
        raise ValueError(f"unknown structure token: {name!r}")

    def on_newline(self, role: str, loss: int) -> None:
        if not self.expecting_newline:
            raise ValueError("newline span is only allowed after message_end")
        if self.current_role is None:
            raise ValueError("newline span appears outside message framing")
        if role != self.current_role:
            raise ValueError("newline span must keep the message role")
        if self.current_loss != loss:
            raise ValueError("loss_mask must be constant across the trailing message newline")
        self.current_role = None
        self.current_loss = None
        self.expecting_newline = False

    def finish(self) -> None:
        if self.expecting_role:
            raise ValueError("message_start is not followed by a role span")
        if self.expecting_newline:
            raise ValueError("message_end is not followed by a trailing newline")
        if self.current_role is not None:
            raise ValueError("message is not closed before end of sequence")
        if self.structure_stack:
            raise ValueError(f"span trace ended with unclosed structure: {self.structure_stack[-1]}")


def _validate_encoded_context_tokens(
    encoded: EncodedContext,
    policy: AgenticContextPolicy,
    layout: _ContextLayout,
) -> tuple[Span, ...]:
    return _walk_encoded_spans(encoded, policy, layout)


def _walk_encoded_spans(
    encoded: EncodedContext,
    policy: AgenticContextPolicy,
    layout: _ContextLayout,
) -> tuple[Span, ...]:
    _validate_encoding_version(encoded.encoding_version)
    if len(encoded.input_ids) != len(encoded.loss_mask):
        raise ValueError("loss_mask length must match input_ids length")
    if any(mask not in {0, 1} for mask in encoded.loss_mask):
        raise ValueError("loss_mask values must be 0 or 1")

    reserved_ids = policy.reserved_ids()
    message_start_id = layout.structure_ids[STRUCTURE_MESSAGE_START]
    newline_ids = layout.newline_ids
    grammar = _GrammarMachine(policy)
    spans: list[Span] = []

    position = 0
    while position < len(encoded.input_ids):
        if encoded.input_ids[position] != message_start_id:
            raise ValueError(f"expected message_start at position {position}")
        message_start_position = position
        message_loss = encoded.loss_mask[position]
        position += 1

        matched_role = None
        for role, ids in layout.ordered_role_prefixes:
            end = position + len(ids)
            if encoded.input_ids[position:end] == ids:
                matched_role = role
                if any(mask != message_loss for mask in encoded.loss_mask[position:end]):
                    raise ValueError(f"loss_mask must be constant within message role prefix for role {role!r}")
                grammar.on_message_start(role, message_loss)
                grammar.on_role(role, message_loss)
                spans.append(
                    Span(
                        start=message_start_position,
                        end=message_start_position + 1,
                        role=role,
                        kind=SPAN_KIND_STRUCTURE,
                    )
                )
                spans.append(Span(start=position, end=end, role=role, kind=SPAN_KIND_ROLE))
                position = end
                break
        if matched_role is None:
            raise ValueError(f"message_start at position {position - 1} is not followed by a valid role prefix")

        while True:
            if position >= len(encoded.input_ids):
                raise ValueError("message is not closed before end of sequence")
            token_id = encoded.input_ids[position]
            if encoded.loss_mask[position] != message_loss:
                raise ValueError("loss_mask must be constant within a message")
            if token_id not in reserved_ids:
                next_position = position + 1
                while next_position < len(encoded.input_ids) and encoded.input_ids[next_position] not in reserved_ids:
                    if encoded.loss_mask[next_position] != message_loss:
                        raise ValueError("loss_mask must be constant within a message")
                    next_position += 1
                kind = (
                    SPAN_KIND_OPAQUE_PAYLOAD
                    if grammar.structure_stack and grammar.structure_stack[-1] == STRUCTURE_OPAQUE_PAYLOAD_START
                    else SPAN_KIND_TEXT
                )
                grammar.on_text(kind, matched_role, message_loss)
                spans.append(Span(start=position, end=next_position, role=matched_role, kind=kind))
                position = next_position
                continue

            name = layout.id_to_structure_name.get(token_id)
            if name is None:
                raise ValueError(f"reserved token id at position {position} is not defined by the policy")
            if name == STRUCTURE_MESSAGE_START:
                raise ValueError(f"nested message_start at position {position}")
            if name == STRUCTURE_MESSAGE_END:
                try:
                    grammar.on_structure(name, matched_role, message_loss)
                except ValueError as exc:
                    if str(exc).startswith("message_end cannot close while nested structure is open"):
                        raise ValueError(
                            f"message_end cannot close while nested structure is open: {grammar.structure_stack[-1]}"
                        ) from exc
                    raise
                spans.append(Span(start=position, end=position + 1, role=matched_role, kind=SPAN_KIND_STRUCTURE))
                position += 1
                newline_end = position + len(newline_ids)
                if encoded.input_ids[position:newline_end] != newline_ids:
                    raise ValueError(f"message_end at position {position - 1} is not followed by the trailing newline")
                if any(mask != message_loss for mask in encoded.loss_mask[position:newline_end]):
                    raise ValueError("loss_mask must be constant across the trailing message newline")
                grammar.on_newline(matched_role, message_loss)
                spans.append(Span(start=position, end=newline_end, role=matched_role, kind=SPAN_KIND_NEWLINE))
                position = newline_end
                break
            try:
                grammar.on_structure(name, matched_role, message_loss)
            except ValueError as exc:
                message = str(exc)
                if message == "opaque_payload_end does not match current structure stack":
                    raise ValueError(f"opaque_payload_end does not match current structure stack at position {position}") from exc
                if message == "structured_region_end does not match current structure stack":
                    raise ValueError(f"structured_region_end does not match current structure stack at position {position}") from exc
                if message.startswith("unknown structure token:"):
                    raise ValueError(f"unknown structure token at position {position}: {name!r}") from exc
                raise
            spans.append(Span(start=position, end=position + 1, role=matched_role, kind=SPAN_KIND_STRUCTURE))
            position += 1

    grammar.finish()
    return tuple(spans)


def _validate_debug_context_spans(
    debug_encoded: DebugEncodedContext,
    policy: AgenticContextPolicy,
    layout: _ContextLayout,
) -> tuple[Span, ...]:
    encoded = debug_encoded.encoded
    spans = debug_encoded.spans
    reserved_ids = policy.reserved_ids()
    expected_start = 0
    allowed_kinds = {
        SPAN_KIND_STRUCTURE,
        SPAN_KIND_ROLE,
        SPAN_KIND_NEWLINE,
        SPAN_KIND_TEXT,
        SPAN_KIND_OPAQUE_PAYLOAD,
    }

    for span in spans:
        if span.start != expected_start:
            raise ValueError(f"spans must be contiguous and ordered: {span}")
        if span.start < 0 or span.end < span.start or span.end > len(encoded.input_ids):
            raise ValueError(f"invalid span range: {span}")
        if span.start == span.end:
            raise ValueError(f"span must not be empty: {span}")
        expected_start = span.end
        if span.role not in policy._allowed_roles:
            raise ValueError(f"span role must be one of the allowed roles: {span}")
        if span.kind not in allowed_kinds:
            raise ValueError(f"unsupported span kind: {span}")

        token_slice = encoded.input_ids[span.start : span.end]

        if span.kind == SPAN_KIND_STRUCTURE:
            if span.end != span.start + 1:
                raise ValueError(f"structure span must cover exactly one token: {span}")
            token_id = token_slice[0]
            if token_id not in reserved_ids:
                raise ValueError(f"structure span does not point to a reserved id: {span}")
            if layout.id_to_structure_name.get(token_id) is None:
                raise ValueError(f"structure span token id is not defined by the policy: {span}")
            continue

        if any(token_id in reserved_ids for token_id in token_slice):
            raise ValueError(f"non-structure span contains reserved id: {span}")
        span_loss = encoded.loss_mask[span.start]
        if any(mask != span_loss for mask in encoded.loss_mask[span.start : span.end]):
            raise ValueError(f"loss_mask values must stay constant within each debug span: {span}")

        if span.kind == SPAN_KIND_ROLE:
            if token_slice != layout.role_prefix_ids[span.role]:
                raise ValueError(f"role span does not match the configured role prefix encoding: {span}")
            continue

        if span.kind == SPAN_KIND_NEWLINE:
            if token_slice != layout.newline_ids:
                raise ValueError(f"newline span does not match the configured newline encoding: {span}")
            continue

    if expected_start != len(encoded.input_ids):
        raise ValueError("spans do not cover the full input sequence")
    _validate_encoding_version(encoded.encoding_version)
    canonical_spans = _canonicalize_debug_spans(spans)
    if canonical_spans != spans:
        raise ValueError("debug spans must be canonical grammar segments")
    return canonical_spans


def _canonicalize_debug_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    if not spans:
        return ()
    canonical: list[Span] = [spans[0]]
    for span in spans[1:]:
        previous = canonical[-1]
        if (
            span.start == previous.end
            and span.role == previous.role
            and span.kind == previous.kind
            and span.kind != SPAN_KIND_STRUCTURE
        ):
            canonical[-1] = Span(
                start=previous.start,
                end=span.end,
                role=previous.role,
                kind=previous.kind,
            )
            continue
        canonical.append(span)
    return tuple(canonical)


def _ensure_tokenizer_matches_policy(
    tokenizer: _TokenizerLike,
    policy: AgenticContextPolicy,
) -> None:
    token_ids = policy.token_table._id_by_name
    token_text = policy.token_table._text_by_name
    for name, expected_id in token_ids.items():
        encoded = tokenizer.encode(token_text[name], add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(
                f"tokenizer does not match policy token table for {name!r}: "
                f"expected {[expected_id]}, got {encoded}"
            )
