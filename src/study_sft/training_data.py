"""Training-specific encoding helpers built on top of normalized conversations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from study_sft.agentic_context import (
    AgenticContextEncoder,
    EncodedContext,
    MessageSpan,
    OpaquePayloadSpan,
)
from study_sft.samples import (
    DatasetFormat,
    TrainingSample,
    agentic_context_from_sample,
    conversation_from_record,
    training_samples_from_conversation,
)


TrainingLabelPolicy = Literal["message", "payload_only"]


@dataclass(frozen=True)
class TrainingEncodingConfig:
    dataset_format: DatasetFormat
    default_belief_prompt: str
    max_length: int
    label_policy: TrainingLabelPolicy = "message"


@dataclass(frozen=True)
class PreparedTrainingEncoding:
    encoded: EncodedContext
    message_spans: tuple[MessageSpan, ...]
    opaque_payload_spans: tuple[OpaquePayloadSpan, ...] = ()


def _has_supervised_labels(labels: list[int]) -> bool:
    return any(label != -100 for label in labels)


def _labels_from_message_loss(encoded: EncodedContext) -> list[int]:
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
    if label_policy == "message":
        return _labels_from_message_loss(prepared.encoded)
    if label_policy == "payload_only":
        return _labels_from_payload_only(prepared)
    raise ValueError(f"unsupported label_policy: {label_policy!r}")


def _truncate_to_supervised_suffix_start(
    encoded: EncodedContext,
    message_spans: tuple[MessageSpan, ...],
    *,
    max_length: int,
) -> int:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(encoded.input_ids) <= max_length:
        return 0

    supervised_indexes = [index for index, span in enumerate(message_spans) if span.loss]
    if not supervised_indexes:
        raise ValueError("training sample contains no supervised messages")

    first_supervised_index = supervised_indexes[0]
    for start_index in range(first_supervised_index + 1):
        start = message_spans[start_index].start
        if len(encoded.input_ids) - start <= max_length:
            return start

    raise ValueError("supervised message suffix exceeds max_length")


def _slice_encoded_context(encoded: EncodedContext, *, start: int) -> EncodedContext:
    return EncodedContext(
        input_ids=encoded.input_ids[start:],
        loss_mask=encoded.loss_mask[start:],
        encoding_version=encoded.encoding_version,
    )


def _slice_message_spans(message_spans: tuple[MessageSpan, ...], *, start: int) -> tuple[MessageSpan, ...]:
    sliced: list[MessageSpan] = []
    for span in message_spans:
        if span.end <= start:
            continue
        sliced.append(
            MessageSpan(
                start=span.start - start,
                end=span.end - start,
                role=span.role,
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
                role=span.role,
                loss=span.loss,
            )
        )
    return tuple(sliced)


def _prepare_training_encoding(
    sample: TrainingSample,
    encoder: AgenticContextEncoder,
    *,
    validate_encoding: bool = False,
    label_policy: TrainingLabelPolicy = "message",
) -> PreparedTrainingEncoding:
    context = agentic_context_from_sample(sample, mark_target_loss=True)
    artifacts = encoder.encode_context_artifacts(
        context,
        include_opaque_payload_spans=label_policy == "payload_only",
        validate=validate_encoding,
    )
    return PreparedTrainingEncoding(
        encoded=artifacts.encoded,
        message_spans=artifacts.message_spans,
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
        prepared.message_spans,
        max_length=max_length,
    )
    if start == 0:
        return prepared
    return PreparedTrainingEncoding(
        encoded=_slice_encoded_context(prepared.encoded, start=start),
        message_spans=_slice_message_spans(prepared.message_spans, start=start),
        opaque_payload_spans=_slice_opaque_payload_spans(prepared.opaque_payload_spans, start=start),
    )


def encode_training_sample(
    sample: TrainingSample,
    encoder: AgenticContextEncoder,
    *,
    max_length: int | None = None,
    validate_encoding: bool = False,
    label_policy: TrainingLabelPolicy = "message",
) -> dict[str, list[int]]:
    prepared = _prepare_training_encoding(
        sample,
        encoder,
        validate_encoding=validate_encoding,
        label_policy=label_policy,
    )
    prepared = _truncate_prepared_training_encoding(prepared, max_length=max_length)
    features = {
        "input_ids": prepared.encoded.input_ids,
        "labels": _labels_from_prepared_training_encoding(
            prepared,
            label_policy=label_policy,
        ),
    }

    if not _has_supervised_labels(features["labels"]):
        raise ValueError("training features contain no supervised labels")
    return features


def iter_training_features_from_record(
    record: dict[str, Any],
    *,
    encoder: AgenticContextEncoder,
    config: TrainingEncodingConfig,
    validate_encoding: bool = False,
):
    conversation = conversation_from_record(
        record,
        config.dataset_format,
        default_belief_prompt=config.default_belief_prompt,
    )
    for sample in training_samples_from_conversation(conversation):
        yield encode_training_sample(
            sample,
            encoder,
            max_length=config.max_length,
            validate_encoding=validate_encoding,
            label_policy=config.label_policy,
        )
