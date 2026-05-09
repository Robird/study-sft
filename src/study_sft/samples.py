"""Normalized conversation builders shared by training, preview, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from study_sft.agentic_context_model import AgenticContext, AgenticMessage, AgenticOpaquePayload, AgenticRole


DatasetFormat = Literal["alpaca", "messages", "sharegpt"]
DATASET_FORMAT_CHOICES: tuple[DatasetFormat, ...] = ("alpaca", "messages", "sharegpt")
NormalizedRole = Literal["system", "user", "assistant"]

DEFAULT_BELIEF_PROMPT = "You are a helpful, honest, and concise assistant."


@dataclass(frozen=True)
class NormalizedTurn:
    role: NormalizedRole
    content: str


@dataclass(frozen=True)
class NormalizedConversation:
    turns: tuple[NormalizedTurn, ...]


@dataclass(frozen=True)
class TrainingSample:
    conversation: NormalizedConversation
    prefix_turn_count: int | None = None
    target_turn_index: int | None = None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = _clean(record.get(key))
        if value:
            return value
    return ""


def _normalize_role(role: Any) -> NormalizedRole:
    normalized = _clean(role).lower()
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    mapped = role_map.get(normalized)
    if mapped is None:
        raise ValueError(f"unsupported conversation role: {role!r}")
    return mapped


def _normalize_message(message: dict[str, Any]) -> NormalizedTurn | None:
    role = _normalize_role(message.get("role") or message.get("from"))
    content = _clean(message.get("content") or message.get("value"))
    if not content:
        return None
    return NormalizedTurn(role=role, content=content)


def conversation_from_record(
    record: dict[str, Any],
    dataset_format: DatasetFormat,
    *,
    default_belief_prompt: str = DEFAULT_BELIEF_PROMPT,
) -> NormalizedConversation:
    if dataset_format == "alpaca":
        return _normalize_alpaca_conversation(record, default_belief_prompt)
    if dataset_format == "messages":
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("messages format expects a list column named 'messages'")
        return _normalize_message_conversation(raw_messages, default_belief_prompt)
    if dataset_format == "sharegpt":
        raw_messages = record.get("conversations") or record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("sharegpt format expects 'conversations' or 'messages'")
        return _normalize_message_conversation(raw_messages, default_belief_prompt)
    raise ValueError(f"unsupported dataset_format: {dataset_format!r}")


def _normalize_alpaca_conversation(
    record: dict[str, Any],
    default_belief_prompt: str,
) -> NormalizedConversation:
    instruction = _first(record, ["instruction", "prompt", "question", "query"])
    input_text = _first(record, ["input", "context"])
    assistant = _first(record, ["output", "response", "answer", "completion"])
    belief = _first(record, ["system", "system_prompt"]) or default_belief_prompt

    if not instruction:
        raise ValueError("alpaca sample has no instruction/prompt/question/query column")
    if not assistant:
        raise ValueError("alpaca sample has no output/response/answer/completion column")

    user = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
    return NormalizedConversation(
        turns=(
            NormalizedTurn(role="system", content=belief),
            NormalizedTurn(role="user", content=user),
            NormalizedTurn(role="assistant", content=assistant),
        )
    )


def _normalize_message_conversation(
    raw_messages: list[dict[str, Any]],
    default_belief_prompt: str,
) -> NormalizedConversation:
    turns = [turn for turn in (_normalize_message(message) for message in raw_messages) if turn is not None]
    if not turns:
        raise ValueError("messages/sharegpt sample is empty after normalization")

    if all(turn.role != "system" for turn in turns):
        turns = [NormalizedTurn(role="system", content=default_belief_prompt), *turns]

    return NormalizedConversation(turns=tuple(turns))


def training_samples_from_conversation(
    conversation: NormalizedConversation,
) -> tuple[TrainingSample, ...]:
    target_turn_indexes = tuple(index for index, turn in enumerate(conversation.turns) if turn.role == "assistant")
    if not target_turn_indexes:
        raise ValueError("messages/sharegpt sample has no assistant target turn")

    return tuple(
        TrainingSample(
            conversation=conversation,
            prefix_turn_count=target_turn_index + 1,
            target_turn_index=target_turn_index,
        )
        for target_turn_index in target_turn_indexes
    )


def _context_role_for_turn(turn: NormalizedTurn) -> AgenticRole:
    return {
        "system": "belief",
        "user": "observation",
        "assistant": "me",
    }[turn.role]


def _validated_prefix_turn_count(conversation: NormalizedConversation, prefix_turn_count: int | None) -> int:
    turn_count = len(conversation.turns)
    if prefix_turn_count is None:
        return turn_count
    if isinstance(prefix_turn_count, bool) or not isinstance(prefix_turn_count, int):
        raise ValueError("prefix_turn_count must be an int when provided")
    if prefix_turn_count < 0 or prefix_turn_count > turn_count:
        raise ValueError(
            f"prefix_turn_count must be between 0 and the conversation length ({turn_count}), got {prefix_turn_count}"
        )
    return prefix_turn_count


def _validated_target_turn_index(
    conversation: NormalizedConversation,
    *,
    prefix_turn_count: int,
    target_turn_index: int | None,
) -> int | None:
    if target_turn_index is None:
        return None
    if isinstance(target_turn_index, bool) or not isinstance(target_turn_index, int):
        raise ValueError("supervision target turn index must be an int when provided")
    if target_turn_index < 0 or target_turn_index >= len(conversation.turns):
        raise ValueError(f"supervision target turn index out of range: {target_turn_index}")
    if target_turn_index >= prefix_turn_count:
        raise ValueError("supervision target turn index must be inside the prefix")
    if conversation.turns[target_turn_index].role != "assistant":
        raise ValueError("supervision target turn index must point to an assistant turn")
    return target_turn_index


def _prefix_turns(conversation: NormalizedConversation, prefix_turn_count: int | None) -> tuple[NormalizedTurn, ...]:
    return conversation.turns[: _validated_prefix_turn_count(conversation, prefix_turn_count)]


def _context_message_for_turn(turn: NormalizedTurn, *, loss: bool = False) -> AgenticMessage:
    return AgenticMessage(
        role=_context_role_for_turn(turn),
        content=(AgenticOpaquePayload(text=turn.content),),
        loss=loss,
    )


def agentic_context_from_conversation(
    conversation: NormalizedConversation,
    *,
    prefix_turn_count: int | None = None,
    target_turn_index: int | None = None,
) -> AgenticContext:
    validated_prefix_turn_count = _validated_prefix_turn_count(conversation, prefix_turn_count)
    validated_target_turn_index = _validated_target_turn_index(
        conversation,
        prefix_turn_count=validated_prefix_turn_count,
        target_turn_index=target_turn_index,
    )
    return AgenticContext(
        messages=tuple(
            _context_message_for_turn(turn, loss=index == validated_target_turn_index)
            for index, turn in enumerate(_prefix_turns(conversation, validated_prefix_turn_count))
        )
    )


def agentic_context_from_sample(
    sample: TrainingSample,
    *,
    mark_target_loss: bool = False,
) -> AgenticContext:
    return agentic_context_from_conversation(
        sample.conversation,
        prefix_turn_count=sample.prefix_turn_count,
        target_turn_index=sample.target_turn_index if mark_target_loss else None,
    )


def conversation_from_user_text(
    user_text: str,
    *,
    belief_prompt: str = DEFAULT_BELIEF_PROMPT,
) -> NormalizedConversation:
    return NormalizedConversation(
        turns=(
            NormalizedTurn(role="system", content=belief_prompt),
            NormalizedTurn(role="user", content=user_text),
        )
    )
