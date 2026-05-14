"""Inference-time helpers for single-turn agentic-context generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from study_sft.agentic_context import AgenticContextEncoder
from study_sft.inference_prompts import InferencePromptConfig, agentic_context_from_user_text
from study_sft.loaders import get_effective_pad_token_id, load_base_tokenizer

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class GenerationInputs:
    prefix_ids: list[int]
    stop_token_ids: list[int]
    pad_token_id: int


STOP_REASON_ENTRY_END = "entry_end"
STOP_REASON_EOS_TOKEN = "eos_token"
STOP_REASON_STRUCTURE_TOKEN = "structure_token"
STOP_REASON_PROTOCOL_VIOLATION = "protocol_violation"
STOP_REASON_MAX_NEW_TOKENS = "max_new_tokens"

GENERATION_MODE_CONTENT = "content"
GENERATION_MODE_ENTRY = "entry"

PARSER_STATE_IN_PAYLOAD = "in_payload"
PARSER_STATE_IN_ACTION = "in_action"
PARSER_STATE_IN_CONTENT = "in_content"
PARSER_STATE_EXPECT_ENTRY_START = "expect_entry_start"
PARSER_STATE_EXPECT_KIND_PREFIX = "expect_kind_prefix"


@dataclass(frozen=True)
class SingleTurnGenerationResult:
    text: str
    display_text: str
    output_ids: list[int]
    content_ids: list[int]
    stop_reason: str
    clean_termination: bool
    stop_token_id: int | None = None
    stop_token_name: str | None = None
    parser_state_at_stop: str = PARSER_STATE_IN_PAYLOAD
    termination_detail: str | None = None


def parse_single_turn_generation(
    output_ids: list[int],
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
    generation_mode: str = GENERATION_MODE_CONTENT,
    next_kind: str = "me",
) -> SingleTurnGenerationResult:
    if generation_mode == GENERATION_MODE_CONTENT:
        return _parse_single_turn_generation_from_content(
            output_ids,
            encoder=encoder,
            tokenizer=tokenizer,
        )
    if generation_mode == GENERATION_MODE_ENTRY:
        return _parse_single_turn_generation_from_entry(
            output_ids,
            encoder=encoder,
            tokenizer=tokenizer,
            next_kind=next_kind,
        )
    raise ValueError(f"unsupported generation_mode: {generation_mode!r}")


def _parse_single_turn_generation_from_content(
    output_ids: list[int],
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
) -> SingleTurnGenerationResult:
    return _parse_single_turn_generation_mixed_content(
        output_ids,
        encoder=encoder,
        tokenizer=tokenizer,
    )


def _parse_single_turn_generation_mixed_content(
    output_ids: list[int],
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
) -> SingleTurnGenerationResult:
    token_table = encoder.policy.token_table
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    content_ids: list[int] = []
    stop_reason = STOP_REASON_MAX_NEW_TOKENS
    clean_termination = False
    stop_token_id: int | None = None
    stop_token_name: str | None = None
    termination_detail: str | None = None
    parser_state = PARSER_STATE_IN_CONTENT
    stack: list[str] = []

    for token_id in output_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = STOP_REASON_EOS_TOKEN
            stop_token_id = token_id
            termination_detail = "eos_before_content_closed" if stack else "eos_before_entry_end"
            break
        if token_id == token_table.entry_end and not stack:
            stop_reason = STOP_REASON_ENTRY_END
            stop_token_id = token_id
            stop_token_name = token_table.name_for_id(token_id)
            termination_detail = "closed_entry"
            clean_termination = True
            break
        if token_id == token_table.entry_start:
            stop_reason = STOP_REASON_STRUCTURE_TOKEN
            stop_token_id = token_id
            stop_token_name = token_table.name_for_id(token_id)
            termination_detail = "unexpected_structure_in_content"
            break
        if token_id == token_table.opaque_payload_start:
            if stack and stack[-1] == "payload":
                stop_reason = STOP_REASON_PROTOCOL_VIOLATION
                stop_token_id = token_id
                stop_token_name = token_table.name_for_id(token_id)
                termination_detail = "unexpected_payload_start_in_payload"
                break
            stack.append("payload")
            content_ids.append(token_id)
            parser_state = PARSER_STATE_IN_PAYLOAD
            continue
        if token_id == token_table.opaque_payload_end:
            if not stack or stack[-1] != "payload":
                stop_reason = STOP_REASON_PROTOCOL_VIOLATION
                stop_token_id = token_id
                stop_token_name = token_table.name_for_id(token_id)
                termination_detail = "unexpected_payload_end_in_content"
                break
            stack.pop()
            content_ids.append(token_id)
            parser_state = _parser_state_for_stack(stack)
            continue
        if token_id == token_table.action_start:
            if stack:
                stop_reason = STOP_REASON_PROTOCOL_VIOLATION
                stop_token_id = token_id
                stop_token_name = token_table.name_for_id(token_id)
                termination_detail = (
                    "unexpected_action_start_in_payload"
                    if stack[-1] == "payload"
                    else "unexpected_nested_action"
                )
                break
            stack.append("action")
            content_ids.append(token_id)
            parser_state = PARSER_STATE_IN_ACTION
            continue
        if token_id == token_table.action_end:
            if not stack or stack[-1] != "action":
                stop_reason = STOP_REASON_PROTOCOL_VIOLATION
                stop_token_id = token_id
                stop_token_name = token_table.name_for_id(token_id)
                termination_detail = "unexpected_action_end_in_content"
                break
            stack.pop()
            content_ids.append(token_id)
            parser_state = _parser_state_for_stack(stack)
            continue
        if token_id == token_table.entry_end:
            stop_reason = STOP_REASON_PROTOCOL_VIOLATION
            stop_token_id = token_id
            stop_token_name = token_table.name_for_id(token_id)
            termination_detail = "entry_end_before_content_closed"
            break
        content_ids.append(token_id)
        parser_state = _parser_state_for_stack(stack)

    if stop_reason == STOP_REASON_MAX_NEW_TOKENS and stack and termination_detail is None:
        termination_detail = "unterminated_content"

    text = tokenizer.decode(content_ids, skip_special_tokens=False)
    return SingleTurnGenerationResult(
        text=text,
        display_text=text.strip(),
        output_ids=list(output_ids),
        content_ids=content_ids,
        stop_reason=stop_reason,
        clean_termination=clean_termination,
        stop_token_id=stop_token_id,
        stop_token_name=stop_token_name,
        parser_state_at_stop=parser_state,
        termination_detail=termination_detail,
    )


def _parser_state_for_stack(stack: list[str]) -> str:
    if not stack:
        return PARSER_STATE_IN_CONTENT
    if stack[-1] == "payload":
        return PARSER_STATE_IN_PAYLOAD
    if stack[-1] == "action":
        return PARSER_STATE_IN_ACTION
    return PARSER_STATE_IN_CONTENT


def _normalize_entry_content_text(
    content_ids: list[int],
    *,
    token_table,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    if len(content_ids) >= 2 and content_ids[0] == token_table.opaque_payload_start:
        if content_ids[-1] == token_table.opaque_payload_end:
            payload_end_index = content_ids.index(token_table.opaque_payload_end)
            if payload_end_index == len(content_ids) - 1:
                return tokenizer.decode(content_ids[1:-1], skip_special_tokens=False)
    return tokenizer.decode(content_ids, skip_special_tokens=False)


def _parse_single_turn_generation_from_entry(
    output_ids: list[int],
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
    next_kind: str,
) -> SingleTurnGenerationResult:
    token_table = encoder.policy.token_table
    stop_reason = STOP_REASON_MAX_NEW_TOKENS
    parser_state = PARSER_STATE_EXPECT_ENTRY_START
    termination_detail: str | None = None
    position = 0
    expected_kind_prefix = encoder._layout_or_build().kind_prefix_ids[next_kind]

    if position >= len(output_ids):
        return SingleTurnGenerationResult(
            text="",
            display_text="",
            output_ids=list(output_ids),
            content_ids=[],
            stop_reason=stop_reason,
            clean_termination=False,
            parser_state_at_stop=parser_state,
            termination_detail=termination_detail,
        )

    if output_ids[position] != token_table.entry_start:
        return SingleTurnGenerationResult(
            text="",
            display_text="",
            output_ids=list(output_ids),
            content_ids=[],
            stop_reason=STOP_REASON_PROTOCOL_VIOLATION,
            clean_termination=False,
            stop_token_id=output_ids[position],
            stop_token_name=token_table.name_for_id(output_ids[position]),
            parser_state_at_stop=parser_state,
            termination_detail="expected_entry_start",
        )
    position += 1
    parser_state = PARSER_STATE_EXPECT_KIND_PREFIX

    actual_prefix = output_ids[position : position + len(expected_kind_prefix)]
    if len(actual_prefix) < len(expected_kind_prefix):
        if actual_prefix == expected_kind_prefix[: len(actual_prefix)]:
            return SingleTurnGenerationResult(
                text="",
                display_text="",
                output_ids=list(output_ids),
                content_ids=[],
                stop_reason=STOP_REASON_MAX_NEW_TOKENS,
                clean_termination=False,
                parser_state_at_stop=parser_state,
                termination_detail=None,
            )
        actual_token_id = output_ids[position] if position < len(output_ids) else None
        return SingleTurnGenerationResult(
            text="",
            display_text="",
            output_ids=list(output_ids),
            content_ids=[],
            stop_reason=STOP_REASON_PROTOCOL_VIOLATION,
            clean_termination=False,
            stop_token_id=actual_token_id,
            stop_token_name=token_table.name_for_id(actual_token_id) if actual_token_id is not None else None,
            parser_state_at_stop=parser_state,
            termination_detail="expected_kind_prefix",
        )

    if actual_prefix != expected_kind_prefix:
        actual_token_id = output_ids[position] if position < len(output_ids) else None
        return SingleTurnGenerationResult(
            text="",
            display_text="",
            output_ids=list(output_ids),
            content_ids=[],
            stop_reason=STOP_REASON_PROTOCOL_VIOLATION,
            clean_termination=False,
            stop_token_id=actual_token_id,
            stop_token_name=token_table.name_for_id(actual_token_id) if actual_token_id is not None else None,
            parser_state_at_stop=parser_state,
            termination_detail="expected_kind_prefix",
        )
    position += len(expected_kind_prefix)
    parser_state = PARSER_STATE_IN_CONTENT

    if position >= len(output_ids):
        return SingleTurnGenerationResult(
            text="",
            display_text="",
            output_ids=list(output_ids),
            content_ids=[],
            stop_reason=STOP_REASON_MAX_NEW_TOKENS,
            clean_termination=False,
            parser_state_at_stop=parser_state,
            termination_detail=None,
        )

    parsed = _parse_single_turn_generation_mixed_content(
        output_ids[position:],
        encoder=encoder,
        tokenizer=tokenizer,
    )
    text = _normalize_entry_content_text(
        parsed.content_ids,
        token_table=token_table,
        tokenizer=tokenizer,
    )
    return SingleTurnGenerationResult(
        text=text,
        display_text=text.strip(),
        output_ids=list(output_ids),
        content_ids=parsed.content_ids,
        stop_reason=parsed.stop_reason,
        clean_termination=parsed.clean_termination,
        stop_token_id=parsed.stop_token_id,
        stop_token_name=parsed.stop_token_name,
        parser_state_at_stop=parsed.parser_state_at_stop,
        termination_detail=parsed.termination_detail,
    )


def format_generation_result(result: SingleTurnGenerationResult) -> str:
    text = result.display_text or "(空输出)"
    if result.clean_termination:
        return text

    status = f"[stop_reason={result.stop_reason}, clean_termination=false"
    if result.termination_detail is not None:
        status += f", detail={result.termination_detail}"
    if result.stop_token_name is not None:
        status += f", token={result.stop_token_name}"
    status += "]"
    return f"{text}\n{status}"


def prepare_single_turn_generation(
    user_text: str,
    encoder: AgenticContextEncoder,
    tokenizer,
    *,
    prompt_config: InferencePromptConfig | None = None,
    generation_mode: str = GENERATION_MODE_CONTENT,
    next_kind: str = "me",
) -> GenerationInputs:
    context = agentic_context_from_user_text(user_text, config=prompt_config)
    if generation_mode == GENERATION_MODE_CONTENT:
        prefix_ids = encoder.encode_generation_entry_prefix(context, next_kind=next_kind)
    elif generation_mode == GENERATION_MODE_ENTRY:
        prefix_ids = encoder.encode_context(context).input_ids
    else:
        raise ValueError(f"unsupported generation_mode: {generation_mode!r}")
    stop_token_ids = [
        token_id
        for token_id in (getattr(tokenizer, "eos_token_id", None), encoder.policy.token_table.entry_end)
        if token_id is not None
    ]
    return GenerationInputs(
        prefix_ids=prefix_ids,
        stop_token_ids=stop_token_ids,
        pad_token_id=get_effective_pad_token_id(tokenizer),
    )


def load_lora_inference_model(
    model_name_or_path: str,
    adapter_path: str,
    *,
    local_files_only: bool,
    load_in_4bit: bool,
) -> tuple[PreTrainedTokenizerBase, PeftModel]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = load_base_tokenizer(
        model_name_or_path,
        local_files_only=local_files_only,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "dtype": dtype,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_single_turn_result(
    user_text: str,
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
    model: PeftModel,
    prompt_config: InferencePromptConfig | None = None,
    generation_mode: str = GENERATION_MODE_CONTENT,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    next_kind: str = "me",
) -> SingleTurnGenerationResult:
    import torch

    generation_inputs = prepare_single_turn_generation(
        user_text,
        encoder,
        tokenizer,
        prompt_config=prompt_config,
        generation_mode=generation_mode,
        next_kind=next_kind,
    )
    device = next(model.parameters()).device
    input_ids = torch.tensor([generation_inputs.prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.long, device=device)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": generation_inputs.pad_token_id,
        "eos_token_id": generation_inputs.stop_token_ids,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

    output_ids = outputs[0][input_ids.shape[-1] :].tolist()
    return parse_single_turn_generation(
        output_ids,
        encoder=encoder,
        tokenizer=tokenizer,
        generation_mode=generation_mode,
        next_kind=next_kind,
    )


def generate_single_turn_text(
    user_text: str,
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
    model: PeftModel,
    prompt_config: InferencePromptConfig | None = None,
    generation_mode: str = GENERATION_MODE_CONTENT,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    next_kind: str = "me",
) -> str:
    return generate_single_turn_result(
        user_text,
        encoder=encoder,
        tokenizer=tokenizer,
        model=model,
        prompt_config=prompt_config,
        generation_mode=generation_mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        next_kind=next_kind,
    ).display_text
