"""Inference-time prompt helpers for single-turn agentic generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from study_sft.agentic_context_model import AgenticContext, AgenticEntry, AgenticKind, AgenticOpaquePayload


NormalizedRole = Literal["system", "user", "assistant"]

DEFAULT_BELIEF_PROMPT = "You are a helpful, honest, and concise assistant."


@dataclass(frozen=True)
class NormalizedTurn:
    role: NormalizedRole
    content: str


@dataclass(frozen=True)
class NormalizedConversation:
    turns: tuple[NormalizedTurn, ...]


def _context_kind_for_turn(turn: NormalizedTurn) -> AgenticKind:
    return {
        "system": "belief",
        "user": "observation",
        "assistant": "me",
    }[turn.role]


def _context_entry_for_turn(turn: NormalizedTurn) -> AgenticEntry:
    return AgenticEntry(
        kind=_context_kind_for_turn(turn),
        content=(AgenticOpaquePayload(text=turn.content),),
    )


def agentic_context_from_conversation(
    conversation: NormalizedConversation,
) -> AgenticContext:
    return AgenticContext(entries=tuple(_context_entry_for_turn(turn) for turn in conversation.turns))


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
