"""Inference-time prompt helpers for single-turn agentic generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html import escape
from typing import Literal

from study_sft.adapters.acml import agentic_context_from_acml_text
from study_sft.agentic_context_model import AgenticContext, AgenticEntry, AgenticKind, AgenticOpaquePayload


NormalizedRole = Literal["system", "user", "assistant"]

DEFAULT_DEVELOPER_NAME = "刘世超"
DEFAULT_MESSAGE_SOURCE = "控制台"
DEFAULT_REPLY_TOOL_NAME = "SendMessage"
DEFAULT_BELIEF_PROMPT = "我应当优先理解开发者的意图，并给出直接、清晰、可执行的回应。"
DEFAULT_INFERENCE_TASK = "single_turn_inference_acml_v1"
BELIEF_ENTITY_REGISTRY_LABEL = "known_people_json:"
DEVELOPER_RELATION = "developer"


@dataclass(frozen=True)
class NormalizedTurn:
    role: NormalizedRole
    content: str


@dataclass(frozen=True)
class NormalizedConversation:
    turns: tuple[NormalizedTurn, ...]


@dataclass(frozen=True)
class InferencePromptConfig:
    belief_prompt: str = DEFAULT_BELIEF_PROMPT
    developer_name: str = DEFAULT_DEVELOPER_NAME
    developer_entity_id: str | None = None
    message_source: str = DEFAULT_MESSAGE_SOURCE
    reply_tool_name: str = DEFAULT_REPLY_TOOL_NAME
    task_name: str = DEFAULT_INFERENCE_TASK


def _context_kind_for_turn(turn: NormalizedTurn) -> AgenticKind:
    if turn.role == "system":
        return "belief"
    if turn.role == "user":
        return "observation"
    return "me"


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


def escape_acml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def escape_acml_attr(text: str) -> str:
    return escape(text, quote=True)


def resolve_developer_entity_id(config: InferencePromptConfig) -> str:
    explicit_entity_id = (config.developer_entity_id or "").strip()
    if explicit_entity_id:
        return explicit_entity_id
    developer_name = config.developer_name.strip() or DEFAULT_DEVELOPER_NAME
    digest = hashlib.sha256(f"developer\x1f{developer_name}".encode("utf-8")).hexdigest()[:10]
    return f"p{digest}"


def build_inference_belief_entity_registry_text(config: InferencePromptConfig) -> str:
    developer_name = config.developer_name.strip() or DEFAULT_DEVELOPER_NAME
    registry = [
        {
            "entity_id": resolve_developer_entity_id(config),
            "entity_type": "person",
            "relation": DEVELOPER_RELATION,
            "name": developer_name,
            "mention": developer_name,
        }
    ]
    return (
        f"{BELIEF_ENTITY_REGISTRY_LABEL}\n"
        f"{json.dumps(registry, ensure_ascii=False, separators=(",", ":"))}\n\n"
    )


def build_inference_belief_text(config: InferencePromptConfig) -> str:
    parts = [
        build_inference_belief_entity_registry_text(config).rstrip(),
        "我当前可调用的对外动作接口有：",
        f"void {config.reply_tool_name}(string target_entity_id, string message);",
        "",
        f"{config.developer_name}是我的开发者。",
    ]
    extra_belief = config.belief_prompt.strip()
    if extra_belief:
        parts.append(extra_belief)
    return "\n".join(parts)


def acml_from_user_text(
    user_text: str,
    *,
    config: InferencePromptConfig | None = None,
) -> str:
    resolved = config or InferencePromptConfig()
    developer_name_text = escape_acml_text(resolved.developer_name)
    developer_name_attr = escape_acml_attr(resolved.developer_name)
    developer_entity_id = escape_acml_attr(resolve_developer_entity_id(resolved))
    message_source = escape_acml_text(resolved.message_source)
    task_name = escape_acml_attr(resolved.task_name)
    observation_payload = escape_acml_text(user_text)
    belief_text = escape_acml_text(build_inference_belief_text(resolved))
    return (
        f'<acml version="0" task="{task_name}" source="infer_lora">'
        '<acml:entry kind="observation" source="console" relation="developer" '
        f'entity_id="{developer_entity_id}" entity_name="{developer_name_attr}" '
        f'entity_mention="{developer_name_attr}">'
        f"我收到{developer_name_text}从{message_source}发来的消息："
        f"<acml:payload>{observation_payload}</acml:payload>"
        "</acml:entry>"
        '<acml:entry kind="belief" source="runtime">'
        f"{belief_text}"
        "</acml:entry>"
        "</acml>"
    )


def agentic_context_from_user_text(
    user_text: str,
    *,
    config: InferencePromptConfig | None = None,
) -> AgenticContext:
    return agentic_context_from_acml_text(
        acml_from_user_text(user_text, config=config),
        loss_policy="none",
    )
