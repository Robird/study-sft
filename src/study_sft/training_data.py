"""ACML-first training encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from study_sft.adapters.acml import agentic_context_from_acml_record
from study_sft.agentic_context import (
    AgenticContextEncoder,
    EncodedContext,
    EntrySpan,
    OpaquePayloadSpan,
)


TrainingLabelPolicy = Literal["entry", "payload_only"]


@dataclass(frozen=True)
class TrainingEncodingConfig:
    max_length: int
    label_policy: TrainingLabelPolicy = "entry"


@dataclass(frozen=True)
class PreparedTrainingEncoding:
    encoded: EncodedContext
    entry_spans: tuple[EntrySpan, ...]
    opaque_payload_spans: tuple[OpaquePayloadSpan, ...] = ()


def _has_supervised_labels(labels: list[int]) -> bool:
    return any(label != -100 for label in labels)


def _labels_from_entry_loss(encoded: EncodedContext) -> list[int]:
    return [
        token_id if loss else -100
        for token_id, loss in zip(encoded.input_ids, encoded.loss_mask, strict=True)
    ]


def _labels_from_payload_only(prepared: PreparedTrainingEncoding) -> list[int]:
    labels = [-100] * len(prepared.encoded.input_ids)
    for span in prepared.opaque_payload_spans:
        if prepared.encoded.loss_mask[span.start] != 1:
            continue
        for index in range(span.start, span.end):
            labels[index] = prepared.encoded.input_ids[index]
    return labels


def _labels_from_prepared_training_encoding(
    prepared: PreparedTrainingEncoding,
    *,
    label_policy: TrainingLabelPolicy,
) -> list[int]:
    if label_policy == "entry":
        return _labels_from_entry_loss(prepared.encoded)
    if label_policy == "payload_only":
        return _labels_from_payload_only(prepared)
    raise ValueError(f"unsupported label_policy: {label_policy!r}")


def _truncate_to_supervised_suffix_start(
    encoded: EncodedContext,
    entry_spans: tuple[EntrySpan, ...],
    *,
    max_length: int,
) -> int:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(encoded.input_ids) <= max_length:
        return 0

    supervised_indexes = [index for index, span in enumerate(entry_spans) if span.loss]
    if not supervised_indexes:
        raise ValueError("training sample contains no supervised entries")

    first_supervised_index = supervised_indexes[0]
    for start_index in range(first_supervised_index + 1):
        start = entry_spans[start_index].start
        if len(encoded.input_ids) - start <= max_length:
            return start

    raise ValueError("supervised entry suffix exceeds max_length")


def _slice_encoded_context(encoded: EncodedContext, *, start: int) -> EncodedContext:
    return EncodedContext(
        input_ids=encoded.input_ids[start:],
        loss_mask=encoded.loss_mask[start:],
        encoding_version=encoded.encoding_version,
    )


def _slice_entry_spans(entry_spans: tuple[EntrySpan, ...], *, start: int) -> tuple[EntrySpan, ...]:
    sliced: list[EntrySpan] = []
    for span in entry_spans:
        if span.end <= start:
            continue
        sliced.append(
            EntrySpan(
                start=span.start - start,
                end=span.end - start,
                kind=span.kind,
                loss=span.loss,
            )
        )
    return tuple(sliced)


def _slice_opaque_payload_spans(
    opaque_payload_spans: tuple[OpaquePayloadSpan, ...],
    *,
    start: int,
) -> tuple[OpaquePayloadSpan, ...]:
    sliced: list[OpaquePayloadSpan] = []
    for span in opaque_payload_spans:
        if span.end <= start:
            continue
        sliced.append(
            OpaquePayloadSpan(
                start=span.start - start,
                end=span.end - start,
                entry_kind=span.entry_kind,
                loss=span.loss,
            )
        )
    return tuple(sliced)


def _prepare_training_context_encoding(
    context,
    encoder: AgenticContextEncoder,
    *,
    validate_encoding: bool = False,
    label_policy: TrainingLabelPolicy = "entry",
) -> PreparedTrainingEncoding:
    artifacts = encoder.encode_context_artifacts(
        context,
        include_opaque_payload_spans=label_policy == "payload_only",
        validate=validate_encoding,
    )
    return PreparedTrainingEncoding(
        encoded=artifacts.encoded,
        entry_spans=artifacts.entry_spans,
        opaque_payload_spans=artifacts.opaque_payload_spans,
    )


def _truncate_prepared_training_encoding(
    prepared: PreparedTrainingEncoding,
    *,
    max_length: int | None,
) -> PreparedTrainingEncoding:
    if max_length is None:
        return prepared
    start = _truncate_to_supervised_suffix_start(
        prepared.encoded,
        prepared.entry_spans,
        max_length=max_length,
    )
    if start == 0:
        return prepared
    return PreparedTrainingEncoding(
        encoded=_slice_encoded_context(prepared.encoded, start=start),
        entry_spans=_slice_entry_spans(prepared.entry_spans, start=start),
        opaque_payload_spans=_slice_opaque_payload_spans(prepared.opaque_payload_spans, start=start),
    )


def encode_training_context(
    context,
    encoder: AgenticContextEncoder,
    *,
    max_length: int | None = None,
    validate_encoding: bool = False,
    label_policy: TrainingLabelPolicy = "entry",
) -> dict[str, list[int]]:
    prepared = _prepare_training_context_encoding(
        context,
        encoder,
        validate_encoding=validate_encoding,
        label_policy=label_policy,
    )
    prepared = _truncate_prepared_training_encoding(prepared, max_length=max_length)
    features = {
        "input_ids": prepared.encoded.input_ids,
        "labels": _labels_from_prepared_training_encoding(prepared, label_policy=label_policy),
    }
    if not _has_supervised_labels(features["labels"]):
        raise ValueError("training features contain no supervised labels")
    return features


def encode_training_features_from_record(
    record: dict[str, Any],
    *,
    encoder: AgenticContextEncoder,
    config: TrainingEncodingConfig,
    validate_encoding: bool = False,
) -> dict[str, list[int]]:
    context = agentic_context_from_acml_record(record)
    return encode_training_context(
        context,
        encoder,
        max_length=config.max_length,
        validate_encoding=validate_encoding,
        label_policy=config.label_policy,
    )
