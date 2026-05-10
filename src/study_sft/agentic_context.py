"""Token-id safe serialization for experimental Agentic context data.

The v0 protocol intentionally stays small:

- entries carry a kind, optional entry-level loss, and a sequence of opaque payload items
- payload items are the only content leaf in the core schema
- generation opens a new entry and optionally the first payload slot explicitly
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from study_sft.agentic_context_model import AgenticContext, AgenticEntry


__all__ = [
    "AgenticContextEncoder",
    "AgenticContextPolicy",
    "AgenticTokenTable",
    "DEFAULT_AGENTIC_CONTEXT_POLICY",
    "DebugEncodedContext",
    "EncodedContext",
    "EncodedContextArtifacts",
    "EncodedText",
    "EntrySpan",
    "OpaquePayloadSpan",
    "QWEN3_AGENTIC_TOKEN_TABLE",
    "Span",
]


ENCODING_VERSION = "agentic-context-v0"

STRUCTURE_MESSAGE_START = "message_start"
STRUCTURE_MESSAGE_END = "message_end"
STRUCTURE_OPAQUE_PAYLOAD_START = "opaque_payload_start"
STRUCTURE_OPAQUE_PAYLOAD_END = "opaque_payload_end"

SPAN_KIND_STRUCTURE = "structure"
SPAN_KIND_KIND = "kind"
SPAN_KIND_NEWLINE = "newline"
SPAN_KIND_OPAQUE_PAYLOAD = "opaque_payload"


class _TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class AgenticTokenTable:
    message_start: int = 151644
    message_end: int = 151645
    opaque_payload_start: int = 151648
    opaque_payload_end: int = 151649

    message_start_text: str = "<|im_start|>"
    message_end_text: str = "<|im_end|>"
    opaque_payload_start_text: str = "<|box_start|>"
    opaque_payload_end_text: str = "<|box_end|>"

    _id_by_name: dict[str, int] = field(init=False, repr=False, compare=False)
    _name_by_id: dict[int, str] = field(init=False, repr=False, compare=False)
    _text_by_name: dict[str, str] = field(init=False, repr=False, compare=False)
    _reserved_ids: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        id_by_name = {
            STRUCTURE_MESSAGE_START: self.message_start,
            STRUCTURE_MESSAGE_END: self.message_end,
            STRUCTURE_OPAQUE_PAYLOAD_START: self.opaque_payload_start,
            STRUCTURE_OPAQUE_PAYLOAD_END: self.opaque_payload_end,
        }
        text_by_name = {
            STRUCTURE_MESSAGE_START: self.message_start_text,
            STRUCTURE_MESSAGE_END: self.message_end_text,
            STRUCTURE_OPAQUE_PAYLOAD_START: self.opaque_payload_start_text,
            STRUCTURE_OPAQUE_PAYLOAD_END: self.opaque_payload_end_text,
        }
        object.__setattr__(self, "_id_by_name", id_by_name)
        object.__setattr__(self, "_name_by_id", {token_id: name for name, token_id in id_by_name.items()})
        object.__setattr__(self, "_text_by_name", text_by_name)
        object.__setattr__(self, "_reserved_ids", frozenset(id_by_name.values()))

    def id_by_name(self) -> dict[str, int]:
        return dict(self._id_by_name)

    def text_by_name(self) -> dict[str, str]:
        return dict(self._text_by_name)

    def name_for_id(self, token_id: int) -> str | None:
        return self._name_by_id.get(token_id)

    def reserved_ids(self) -> frozenset[int]:
        return self._reserved_ids


QWEN3_AGENTIC_TOKEN_TABLE = AgenticTokenTable()


@dataclass(frozen=True, slots=True)
class AgenticContextPolicy:
    token_table: AgenticTokenTable = QWEN3_AGENTIC_TOKEN_TABLE
    allowed_kinds: tuple[str, ...] = ("belief", "observation", "me")
    extra_reserved_ids: tuple[int, ...] = ()

    _reserved_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    _allowed_kinds: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_reserved_ids",
            frozenset((*self.token_table.reserved_ids(), *self.extra_reserved_ids)),
        )
        object.__setattr__(self, "_allowed_kinds", frozenset(self.allowed_kinds))

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
    entry_kind: str
    kind: str


@dataclass(frozen=True, slots=True)
class EntrySpan:
    start: int
    end: int
    kind: str
    loss: bool


@dataclass(frozen=True, slots=True)
class OpaquePayloadSpan:
    start: int
    end: int
    entry_kind: str
    loss: bool


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
    entry_spans: tuple[EntrySpan, ...] = ()
    opaque_payload_spans: tuple[OpaquePayloadSpan, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded": self.encoded.to_dict(),
            "spans": [asdict(span) for span in self.spans],
            "entry_spans": [asdict(span) for span in self.entry_spans],
            "opaque_payload_spans": [asdict(span) for span in self.opaque_payload_spans],
        }


@dataclass(frozen=True, slots=True)
class EncodedContextArtifacts:
    encoded: EncodedContext
    entry_spans: tuple[EntrySpan, ...] = ()
    opaque_payload_spans: tuple[OpaquePayloadSpan, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded": self.encoded.to_dict(),
            "entry_spans": [asdict(span) for span in self.entry_spans],
            "opaque_payload_spans": [asdict(span) for span in self.opaque_payload_spans],
        }


@dataclass(frozen=True, slots=True)
class _NormalizedEntry:
    kind: str
    loss: bool
    content: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedContext:
    entries: tuple[_NormalizedEntry, ...]


@dataclass(frozen=True, slots=True)
class _ContextLayout:
    structure_ids: dict[str, int]
    id_to_structure_name: dict[int, str]
    kind_prefix_ids: dict[str, list[int]]
    ordered_kind_prefixes: tuple[tuple[str, list[int]], ...]
    newline_ids: list[int]


def _escape_opaque_payload_text(text: str) -> str:
    replacements = {
        "\\": "\\\\",
        "<": "\\u003c",
        ">": "\\u003e",
    }
    return "".join(replacements.get(char, char) for char in text)


def _unescape_opaque_payload_text(text: str) -> str:
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


def _contains_reserved_id(input_ids: list[int], reserved_ids: frozenset[int]) -> bool:
    return any(token_id in reserved_ids for token_id in input_ids)


def _validate_encoding_version(encoding_version: str) -> None:
    if encoding_version != ENCODING_VERSION:
        raise ValueError(f"unsupported encoding_version: {encoding_version!r}; expected {ENCODING_VERSION!r}")


def _normalize_context(
    context: AgenticContext,
    policy: AgenticContextPolicy,
) -> _NormalizedContext:
    if not isinstance(context, AgenticContext):
        raise ValueError("context must be an AgenticContext; parse external dicts with agentic_context_schema first")
    return _NormalizedContext(entries=tuple(_normalize_typed_entry(entry, policy) for entry in context.entries))


def _normalize_typed_entry(
    entry: AgenticEntry,
    policy: AgenticContextPolicy,
) -> _NormalizedEntry:
    kind = _validate_kind(entry.kind, policy)
    return _NormalizedEntry(
        kind=kind,
        loss=entry.loss,
        content=tuple(item.text for item in entry.content),
    )


def _validate_kind(kind_value: Any, policy: AgenticContextPolicy) -> str:
    if not isinstance(kind_value, str):
        raise ValueError("entry.kind must be a string")
    kind = kind_value.strip()
    if not kind:
        raise ValueError("entry.kind must not be empty")
    if kind not in policy._allowed_kinds:
        raise ValueError(f"unsupported kind: {kind!r}")
    return kind


class AgenticContextEncoder:
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
        raw_text = "" if text is None else str(text)
        escaped_text = _escape_opaque_payload_text(raw_text)
        escaped_ids = self.tokenizer.encode(escaped_text, add_special_tokens=False)
        if _contains_reserved_id(escaped_ids, self.reserved_ids):
            raise ValueError("escaped untrusted text still produced reserved token ids")
        return EncodedText(input_ids=escaped_ids, encoding="text-escaped", text=escaped_text)

    def encode_context(
        self,
        context: AgenticContext,
        *,
        validate: bool = False,
    ) -> EncodedContext:
        normalized = _normalize_context(context, self.policy)
        encoded = self._encode_artifacts(
            normalized,
            collect_debug=False,
            collect_entry_spans=False,
            collect_payload_spans=False,
        ).encoded()
        if validate:
            self.validate(encoded)
        return encoded

    def encode_context_artifacts(
        self,
        context: AgenticContext,
        *,
        include_opaque_payload_spans: bool = False,
        validate: bool = False,
    ) -> EncodedContextArtifacts:
        normalized = _normalize_context(context, self.policy)
        artifacts = self._encode_artifacts(
            normalized,
            collect_debug=False,
            collect_entry_spans=True,
            collect_payload_spans=include_opaque_payload_spans,
        ).artifacts()
        if validate:
            self.validate(artifacts.encoded)
        return artifacts

    def encode_context_with_debug(
        self,
        context: AgenticContext,
        *,
        validate: bool = False,
    ) -> DebugEncodedContext:
        normalized = _normalize_context(context, self.policy)
        debug_encoded = self._encode_artifacts(
            normalized,
            collect_debug=True,
            collect_entry_spans=True,
            collect_payload_spans=True,
        ).debug_encoded()
        if validate:
            self.validate_debug(debug_encoded)
        return debug_encoded

    def encode_generation_payload_prefix(
        self,
        context: AgenticContext,
        *,
        next_kind: str = "me",
    ) -> list[int]:
        layout = self._layout_or_build()
        return [
            *self._encode_generation_entry_prefix(context, next_kind=next_kind),
            layout.structure_ids[STRUCTURE_OPAQUE_PAYLOAD_START],
        ]

    def validate(self, encoded: EncodedContext) -> None:
        _walk_encoded_spans(encoded, self.policy, self._layout_or_build())

    def validate_debug(self, debug_encoded: DebugEncodedContext) -> None:
        layout = self._layout_or_build()
        encoded_spans = _walk_encoded_spans(debug_encoded.encoded, self.policy, layout)
        debug_spans = _validate_debug_context_spans(debug_encoded, self.policy, layout)
        if debug_spans != encoded_spans:
            raise ValueError("debug spans do not match the encoded token trace")
        if debug_encoded.entry_spans:
            encoded_entries = _walk_encoded_entries(debug_encoded.encoded, self.policy, layout)
            if debug_encoded.entry_spans != encoded_entries:
                raise ValueError("debug entry_spans do not match the encoded token trace")
        if debug_encoded.opaque_payload_spans:
            encoded_artifacts = _walk_encoded_artifacts(debug_encoded.encoded, self.policy, layout)
            if debug_encoded.opaque_payload_spans != encoded_artifacts.opaque_payload_spans:
                raise ValueError("debug opaque_payload_spans do not match the encoded token trace")

    def describe_entries(self, encoded: EncodedContext) -> tuple[EntrySpan, ...]:
        return _walk_encoded_entries(encoded, self.policy, self._layout_or_build())

    def _encode_artifacts(
        self,
        normalized: _NormalizedContext,
        *,
        collect_debug: bool,
        collect_entry_spans: bool,
        collect_payload_spans: bool,
    ) -> _ContextBuilder:
        builder = _ContextBuilder(
            self,
            collect_debug=collect_debug,
            collect_entry_spans=collect_entry_spans,
            collect_payload_spans=collect_payload_spans,
        )
        for entry in normalized.entries:
            builder.serialize_entry(entry)
        return builder

    def _encode_generation_entry_prefix(
        self,
        context: AgenticContext,
        *,
        next_kind: str,
    ) -> list[int]:
        normalized = _normalize_context(context, self.policy)
        encoded = self._encode_artifacts(
            normalized,
            collect_debug=False,
            collect_entry_spans=False,
            collect_payload_spans=False,
        ).encoded()
        kind = _validate_kind(next_kind, self.policy)
        layout = self._layout_or_build()
        return [
            *encoded.input_ids,
            layout.structure_ids[STRUCTURE_MESSAGE_START],
            *layout.kind_prefix_ids[kind],
        ]

    def _layout_or_build(self) -> _ContextLayout:
        if self._layout is None:
            structure_ids = self.policy.token_table.id_by_name()
            kind_prefix_ids = {
                kind: self._encode_checked_inline(f"{kind}\n", kind=SPAN_KIND_KIND)
                for kind in self.policy.allowed_kinds
            }
            newline_ids = self._encode_checked_inline("\n", kind=SPAN_KIND_NEWLINE)
            self._layout = _ContextLayout(
                structure_ids=structure_ids,
                id_to_structure_name={token_id: name for name, token_id in structure_ids.items()},
                kind_prefix_ids=kind_prefix_ids,
                ordered_kind_prefixes=tuple(
                    sorted(
                        ((kind, list(ids)) for kind, ids in kind_prefix_ids.items()),
                        key=lambda item: len(item[1]),
                        reverse=True,
                    )
                ),
                newline_ids=newline_ids,
            )
        return self._layout

    def _encode_checked_inline(self, text: str, *, kind: str) -> list[int]:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if _contains_reserved_id(input_ids, self.reserved_ids):
            raise ValueError(f"checked-inline text for kind={kind!r} produced reserved token ids")
        return input_ids


class _ContextBuilder:
    def __init__(
        self,
        encoder: AgenticContextEncoder,
        *,
        collect_debug: bool,
        collect_entry_spans: bool = False,
        collect_payload_spans: bool = False,
    ) -> None:
        self.encoder = encoder
        self.layout = encoder._layout_or_build()
        self.collect_debug = collect_debug
        self.collect_entry_spans = collect_entry_spans
        self.collect_payload_spans = collect_payload_spans
        self.input_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.spans: list[Span] = []
        self.entry_spans: list[EntrySpan] = []
        self.opaque_payload_spans: list[OpaquePayloadSpan] = []

    def serialize_entry(self, entry: _NormalizedEntry) -> None:
        start = len(self.input_ids)
        self._append_structure(STRUCTURE_MESSAGE_START, entry_kind=entry.kind, loss=entry.loss)
        self._append_ids(
            self.layout.kind_prefix_ids[entry.kind],
            entry_kind=entry.kind,
            loss=entry.loss,
            span_kind=SPAN_KIND_KIND,
        )
        for payload_text in entry.content:
            self._serialize_payload(payload_text, entry_kind=entry.kind, loss=entry.loss)
        self._append_structure(STRUCTURE_MESSAGE_END, entry_kind=entry.kind, loss=entry.loss)
        self._append_ids(
            self.layout.newline_ids,
            entry_kind=entry.kind,
            loss=entry.loss,
            span_kind=SPAN_KIND_NEWLINE,
        )
        if self.collect_entry_spans:
            self.entry_spans.append(EntrySpan(start=start, end=len(self.input_ids), kind=entry.kind, loss=entry.loss))

    def _serialize_payload(self, payload_text: str, *, entry_kind: str, loss: bool) -> None:
        self._append_structure(STRUCTURE_OPAQUE_PAYLOAD_START, entry_kind=entry_kind, loss=loss)
        payload_start = len(self.input_ids)
        encoded = self.encoder.encode_payload(payload_text)
        self._append_ids(
            encoded.input_ids,
            entry_kind=entry_kind,
            loss=loss,
            span_kind=SPAN_KIND_OPAQUE_PAYLOAD,
        )
        payload_end = len(self.input_ids)
        if self.collect_payload_spans:
            self.opaque_payload_spans.append(
                OpaquePayloadSpan(start=payload_start, end=payload_end, entry_kind=entry_kind, loss=loss)
            )
        self._append_structure(STRUCTURE_OPAQUE_PAYLOAD_END, entry_kind=entry_kind, loss=loss)

    def _append_structure(self, name: str, *, entry_kind: str, loss: bool) -> None:
        token_id = self.layout.structure_ids[name]
        start = len(self.input_ids)
        self.input_ids.append(token_id)
        self.loss_mask.append(1 if loss else 0)
        if self.collect_debug:
            self.spans.append(
                Span(start=start, end=start + 1, entry_kind=entry_kind, kind=SPAN_KIND_STRUCTURE)
            )

    def _append_ids(
        self,
        input_ids: list[int],
        *,
        entry_kind: str,
        loss: bool,
        span_kind: str,
    ) -> None:
        if not input_ids:
            return
        start = len(self.input_ids)
        self.input_ids.extend(input_ids)
        self.loss_mask.extend([1 if loss else 0] * len(input_ids))
        if self.collect_debug:
            self.spans.append(
                Span(start=start, end=len(self.input_ids), entry_kind=entry_kind, kind=span_kind)
            )

    def encoded(self) -> EncodedContext:
        return EncodedContext(input_ids=self.input_ids, loss_mask=self.loss_mask)

    def debug_encoded(self) -> DebugEncodedContext:
        if not self.collect_debug:
            raise RuntimeError("debug trace collection was not enabled")
        return DebugEncodedContext(
            encoded=self.encoded(),
            spans=tuple(self.spans),
            entry_spans=tuple(self.entry_spans),
            opaque_payload_spans=tuple(self.opaque_payload_spans),
        )

    def artifacts(self) -> EncodedContextArtifacts:
        if not self.collect_entry_spans:
            raise RuntimeError("entry span collection was not enabled")
        return EncodedContextArtifacts(
            encoded=self.encoded(),
            entry_spans=tuple(self.entry_spans),
            opaque_payload_spans=tuple(self.opaque_payload_spans),
        )


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

    spans: list[Span] = []
    reserved_ids = policy.reserved_ids()
    position = 0
    while position < len(encoded.input_ids):
        if encoded.input_ids[position] != layout.structure_ids[STRUCTURE_MESSAGE_START]:
            raise ValueError(f"expected message_start at position {position}")
        entry_start = position
        entry_loss = encoded.loss_mask[position]
        position += 1
        matched_kind = None
        for kind, ids in layout.ordered_kind_prefixes:
            end = position + len(ids)
            if encoded.input_ids[position:end] == ids:
                if any(mask != entry_loss for mask in encoded.loss_mask[position:end]):
                    raise ValueError("loss_mask must be constant within entry kind prefix")
                matched_kind = kind
                spans.append(Span(start=entry_start, end=entry_start + 1, entry_kind=kind, kind=SPAN_KIND_STRUCTURE))
                spans.append(Span(start=position, end=end, entry_kind=kind, kind=SPAN_KIND_KIND))
                position = end
                break
        if matched_kind is None:
            raise ValueError(f"message_start at position {entry_start} is not followed by a valid kind prefix")

        while True:
            if position >= len(encoded.input_ids):
                raise ValueError("entry is not closed before end of sequence")
            token_id = encoded.input_ids[position]
            if encoded.loss_mask[position] != entry_loss:
                raise ValueError("loss_mask must be constant within an entry")
            if token_id == layout.structure_ids[STRUCTURE_OPAQUE_PAYLOAD_START]:
                spans.append(Span(start=position, end=position + 1, entry_kind=matched_kind, kind=SPAN_KIND_STRUCTURE))
                position += 1
                payload_start = position
                while position < len(encoded.input_ids) and encoded.input_ids[position] not in reserved_ids:
                    if encoded.loss_mask[position] != entry_loss:
                        raise ValueError("loss_mask must be constant within an entry")
                    position += 1
                if position >= len(encoded.input_ids):
                    raise ValueError("opaque_payload is not closed before end of sequence")
                spans.append(
                    Span(start=payload_start, end=position, entry_kind=matched_kind, kind=SPAN_KIND_OPAQUE_PAYLOAD)
                )
                if encoded.input_ids[position] != layout.structure_ids[STRUCTURE_OPAQUE_PAYLOAD_END]:
                    raise ValueError(f"opaque_payload contains an unexpected reserved token at position {position}")
                spans.append(Span(start=position, end=position + 1, entry_kind=matched_kind, kind=SPAN_KIND_STRUCTURE))
                position += 1
                continue
            if token_id == layout.structure_ids[STRUCTURE_MESSAGE_END]:
                spans.append(Span(start=position, end=position + 1, entry_kind=matched_kind, kind=SPAN_KIND_STRUCTURE))
                position += 1
                newline_end = position + len(layout.newline_ids)
                if encoded.input_ids[position:newline_end] != layout.newline_ids:
                    raise ValueError(f"message_end at position {position - 1} is not followed by the trailing newline")
                if any(mask != entry_loss for mask in encoded.loss_mask[position:newline_end]):
                    raise ValueError("loss_mask must be constant across the trailing entry newline")
                spans.append(Span(start=position, end=newline_end, entry_kind=matched_kind, kind=SPAN_KIND_NEWLINE))
                position = newline_end
                break
            raise ValueError(f"entry content must be opaque_payload blocks, found reserved token at position {position}")
    return tuple(spans)


def _walk_encoded_entries(
    encoded: EncodedContext,
    policy: AgenticContextPolicy,
    layout: _ContextLayout,
) -> tuple[EntrySpan, ...]:
    spans = _walk_encoded_spans(encoded, policy, layout)
    if not spans:
        return ()
    entries: list[EntrySpan] = []
    index = 0
    while index < len(spans):
        start_span = spans[index]
        entry_kind = start_span.entry_kind
        loss = bool(encoded.loss_mask[start_span.start])
        while index < len(spans) and spans[index].kind != SPAN_KIND_NEWLINE:
            index += 1
        if index >= len(spans):
            raise ValueError("entry trace ended before trailing newline")
        entries.append(EntrySpan(start=start_span.start, end=spans[index].end, kind=entry_kind, loss=loss))
        index += 1
    return tuple(entries)


def _walk_encoded_artifacts(
    encoded: EncodedContext,
    policy: AgenticContextPolicy,
    layout: _ContextLayout,
) -> EncodedContextArtifacts:
    entry_spans = _walk_encoded_entries(encoded, policy, layout)
    spans = _walk_encoded_spans(encoded, policy, layout)
    opaque_payload_spans = tuple(
        OpaquePayloadSpan(
            start=span.start,
            end=span.end,
            entry_kind=span.entry_kind,
            loss=bool(encoded.loss_mask[span.start]) if span.start < span.end else False,
        )
        for span in spans
        if span.kind == SPAN_KIND_OPAQUE_PAYLOAD
    )
    return EncodedContextArtifacts(
        encoded=encoded,
        entry_spans=entry_spans,
        opaque_payload_spans=opaque_payload_spans,
    )


def _validate_debug_context_spans(
    debug_encoded: DebugEncodedContext,
    policy: AgenticContextPolicy,
    layout: _ContextLayout,
) -> tuple[Span, ...]:
    encoded = debug_encoded.encoded
    spans = debug_encoded.spans
    expected_start = 0
    allowed_span_kinds = {
        SPAN_KIND_STRUCTURE,
        SPAN_KIND_KIND,
        SPAN_KIND_NEWLINE,
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
        if span.entry_kind not in policy._allowed_kinds:
            raise ValueError(f"span entry kind must be one of the allowed kinds: {span}")
        if span.kind not in allowed_span_kinds:
            raise ValueError(f"unsupported span kind: {span}")
        if span.kind == SPAN_KIND_STRUCTURE and span.end != span.start + 1:
            raise ValueError(f"structure span must cover exactly one token: {span}")
        if span.kind == SPAN_KIND_KIND:
            if encoded.input_ids[span.start:span.end] != layout.kind_prefix_ids[span.entry_kind]:
                raise ValueError(f"kind span does not match the configured kind prefix encoding: {span}")
        if span.kind == SPAN_KIND_NEWLINE:
            if encoded.input_ids[span.start:span.end] != layout.newline_ids:
                raise ValueError(f"newline span does not match the configured newline encoding: {span}")
    if expected_start != len(encoded.input_ids):
        raise ValueError("spans do not cover the full input sequence")
    _validate_encoding_version(encoded.encoding_version)
    return tuple(spans)


def _ensure_tokenizer_matches_policy(tokenizer: _TokenizerLike, policy: AgenticContextPolicy) -> None:
    token_ids = policy.token_table.id_by_name()
    token_text = policy.token_table.text_by_name()
    for name, expected_id in token_ids.items():
        encoded = tokenizer.encode(token_text[name], add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(
                f"tokenizer does not match policy token table for {name!r}: expected {[expected_id]}, got {encoded}"
            )
