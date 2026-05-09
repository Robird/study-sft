"""Inference-time helpers for single-turn agentic-context generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from study_sft.agentic_context import AgenticContextEncoder
from study_sft.loaders import get_effective_pad_token_id, load_base_tokenizer
from study_sft.samples import agentic_context_from_conversation, conversation_from_user_text

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class GenerationInputs:
    prefix_ids: list[int]
    stop_token_ids: list[int]
    pad_token_id: int


STOP_REASON_MESSAGE_END = "message_end"
STOP_REASON_EOS_TOKEN = "eos_token"
STOP_REASON_STRUCTURE_TOKEN = "structure_token"
STOP_REASON_PROTOCOL_VIOLATION = "protocol_violation"
STOP_REASON_MAX_NEW_TOKENS = "max_new_tokens"

PARSER_STATE_IN_PAYLOAD = "in_payload"
PARSER_STATE_AFTER_PAYLOAD_END = "after_payload_end"


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
) -> SingleTurnGenerationResult:
    token_table = encoder.policy.token_table
    reserved_ids = encoder.policy.reserved_ids()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    content_ids: list[int] = []
    stop_reason = STOP_REASON_MAX_NEW_TOKENS
    clean_termination = False
    stop_token_id: int | None = None
    stop_token_name: str | None = None
    parser_state = PARSER_STATE_IN_PAYLOAD
    termination_detail: str | None = None

    for token_id in output_ids:
        if parser_state == PARSER_STATE_IN_PAYLOAD:
            if token_id == token_table.opaque_payload_end:
                parser_state = PARSER_STATE_AFTER_PAYLOAD_END
                termination_detail = "payload_closed"
                continue
            if token_id == token_table.message_end:
                stop_reason = STOP_REASON_MESSAGE_END
                stop_token_id = token_id
                stop_token_name = token_table.name_for_id(token_id)
                termination_detail = "message_end_before_payload_end"
                break
            if eos_token_id is not None and token_id == eos_token_id:
                stop_reason = STOP_REASON_EOS_TOKEN
                stop_token_id = token_id
                termination_detail = "eos_before_payload_end"
                break
            if token_id in reserved_ids:
                stop_reason = STOP_REASON_STRUCTURE_TOKEN
                stop_token_id = token_id
                stop_token_name = token_table.name_for_id(token_id)
                termination_detail = "unexpected_structure_in_payload"
                break
            content_ids.append(token_id)
            continue

        if token_id == token_table.message_end:
            stop_reason = STOP_REASON_MESSAGE_END
            clean_termination = True
            stop_token_id = token_id
            stop_token_name = token_table.name_for_id(token_id)
            termination_detail = "closed_message"
            break
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = STOP_REASON_EOS_TOKEN
            stop_token_id = token_id
            termination_detail = "eos_after_payload_end"
            break
        if token_id in reserved_ids:
            stop_reason = STOP_REASON_STRUCTURE_TOKEN
            stop_token_id = token_id
            stop_token_name = token_table.name_for_id(token_id)
            termination_detail = "unexpected_structure_after_payload_end"
            break
        stop_reason = STOP_REASON_PROTOCOL_VIOLATION
        termination_detail = "text_after_payload_end"
        break

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
    belief_prompt: str,
    next_role: str = "me",
) -> GenerationInputs:
    context = agentic_context_from_conversation(
        conversation_from_user_text(user_text, belief_prompt=belief_prompt)
    )
    prefix_ids = encoder.encode_generation_payload_prefix(context, next_role=next_role)
    stop_token_ids = [
        token_id
        for token_id in (getattr(tokenizer, "eos_token_id", None), encoder.policy.token_table.message_end)
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
        "torch_dtype": dtype,
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
    belief_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    next_role: str = "me",
) -> SingleTurnGenerationResult:
    import torch

    generation_inputs = prepare_single_turn_generation(
        user_text,
        encoder,
        tokenizer,
        belief_prompt=belief_prompt,
        next_role=next_role,
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
    )


def generate_single_turn_text(
    user_text: str,
    *,
    encoder: AgenticContextEncoder,
    tokenizer: PreTrainedTokenizerBase,
    model: PeftModel,
    belief_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    next_role: str = "me",
) -> str:
    return generate_single_turn_result(
        user_text,
        encoder=encoder,
        tokenizer=tokenizer,
        model=model,
        belief_prompt=belief_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        next_role=next_role,
    ).display_text
