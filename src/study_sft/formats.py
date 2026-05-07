"""Prompt/text formatting helpers for SFT experiments.

The key idea of this project is to keep the model, dataset, and training
hyper-parameters stable while swapping only the text protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal


PromptMode = Literal["chatml", "late_system", "bora"]
DatasetFormat = Literal["alpaca", "messages", "sharegpt", "text"]

DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest, and concise assistant."
DEFAULT_BORA_REASONING = "I need to satisfy the observation using the current belief."


@dataclass(frozen=True)
class SftExample:
    system: str
    user: str
    assistant: str
    history: tuple[dict[str, str], ...] = ()
    text: str | None = None


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


def _chatml_message(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n"


def _normalise_role(role: str) -> str:
    role = role.strip().lower()
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    return role_map.get(role, role)


def _normalise_message(message: dict[str, Any]) -> dict[str, str]:
    role = _clean(message.get("role") or message.get("from"))
    content = _clean(message.get("content") or message.get("value"))
    return {"role": _normalise_role(role), "content": content}


def _split_messages(messages: list[dict[str, str]], default_system: str) -> SftExample:
    clean_messages = [message for message in messages if message["role"] and message["content"]]
    if not clean_messages:
        raise ValueError("messages/sharegpt sample is empty after normalization")

    system = default_system
    non_system_messages: list[dict[str, str]] = []
    for message in clean_messages:
        if message["role"] == "system" and system == default_system:
            system = message["content"]
        elif message["role"] != "system":
            non_system_messages.append(message)

    target_index = None
    for index in range(len(non_system_messages) - 1, -1, -1):
        if non_system_messages[index]["role"] == "assistant":
            target_index = index
            break
    if target_index is None:
        raise ValueError("messages/sharegpt sample has no assistant target turn")

    target = non_system_messages[target_index]
    history = tuple(non_system_messages[:target_index])
    user = "\n\n".join(
        f"{message['role']}: {message['content']}" for message in history if message["role"] != "assistant"
    ).strip()
    if not user and history:
        user = history[-1]["content"]
    return SftExample(system=system, user=user, assistant=target["content"], history=history)


def extract_example(
    record: dict[str, Any],
    dataset_format: DatasetFormat,
    default_system: str = DEFAULT_SYSTEM_PROMPT,
) -> SftExample:
    if dataset_format == "text":
        text = _first(record, ["text", "content", "completion"])
        if not text:
            raise ValueError("text sample has no text/content/completion column")
        return SftExample(system=default_system, user="", assistant="", text=text)

    if dataset_format == "messages":
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("messages format expects a list column named 'messages'")
        return _split_messages([_normalise_message(message) for message in raw_messages], default_system)

    if dataset_format == "sharegpt":
        raw_messages = record.get("conversations") or record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("sharegpt format expects 'conversations' or 'messages'")
        return _split_messages([_normalise_message(message) for message in raw_messages], default_system)

    instruction = _first(record, ["instruction", "prompt", "question", "query"])
    input_text = _first(record, ["input", "context"])
    assistant = _first(record, ["output", "response", "answer", "completion"])
    system = _first(record, ["system", "system_prompt"]) or default_system

    if not instruction:
        raise ValueError("alpaca sample has no instruction/prompt/question/query column")
    if not assistant:
        raise ValueError("alpaca sample has no output/response/answer/completion column")

    user = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
    return SftExample(system=system, user=user, assistant=assistant)


def format_sft_text(
    record: dict[str, Any],
    dataset_format: DatasetFormat,
    prompt_mode: PromptMode,
    default_system: str = DEFAULT_SYSTEM_PROMPT,
    bora_reasoning: str = DEFAULT_BORA_REASONING,
) -> str:
    example = extract_example(record, dataset_format, default_system)
    if example.text is not None:
        return example.text

    if prompt_mode == "chatml":
        messages = [{"role": "system", "content": example.system}]
        messages.extend(example.history or ({"role": "user", "content": example.user},))
        messages.append({"role": "assistant", "content": example.assistant})
        return "".join(_chatml_message(message["role"], message["content"]) for message in messages)

    if prompt_mode == "late_system":
        history = list(example.history or ({"role": "user", "content": example.user},))
        prompt = "".join(_chatml_message(message["role"], message["content"]) for message in history)
        prompt += _chatml_message("system", example.system)
        prompt += _chatml_message("assistant", example.assistant)
        return prompt

    return (
        _chatml_message("belief", example.system)
        + _chatml_message("observation", example.user)
        + _chatml_message(
            "assistant",
            f"Reasoning:\n{bora_reasoning.strip()}\n\nAction:\n{example.assistant.strip()}",
        )
    )


def format_generation_prompt(
    user: str,
    prompt_mode: PromptMode,
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    user = user.strip()
    system = system.strip() or DEFAULT_SYSTEM_PROMPT
    if prompt_mode == "chatml":
        return _chatml_message("system", system) + _chatml_message("user", user) + "<|im_start|>assistant\n"
    if prompt_mode == "late_system":
        return _chatml_message("user", user) + _chatml_message("system", system) + "<|im_start|>assistant\n"
    return (
        _chatml_message("belief", system)
        + _chatml_message("observation", user)
        + "<|im_start|>assistant\nReasoning:\n"
    )
