"""Run one-off or interactive single-turn generation with a trained LoRA adapter."""

from __future__ import annotations

import argparse

from study_sft.cli_args import (
    add_belief_prompt_arg,
    add_inference_context_args,
    add_model_source_args,
    add_optional_bool_arg,
)
from study_sft.agentic_context import AgenticContextEncoder
from study_sft.inference_prompts import InferencePromptConfig
from study_sft.inference_runtime import (
    GENERATION_MODE_CONTENT,
    GENERATION_MODE_ENTRY,
    INFERENCE_DEVICE_MAP_AUTO,
    INFERENCE_DEVICE_MAP_CPU,
    INFERENCE_DEVICE_MAP_SINGLE,
    SingleTurnGenerationResult,
    format_generation_result,
    format_generation_token_debug,
    generate_single_turn_result,
    load_lora_inference_model,
)


EXIT_COMMANDS = {"exit", "quit", ":q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_source_args(parser)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--prompt")
    add_optional_bool_arg(
        parser,
        "--interactive",
        default=None,
        help="进入交互式单轮推理；默认在未提供 --prompt 时自动开启",
    )
    add_inference_context_args(parser)
    add_belief_prompt_arg(parser)
    parser.add_argument(
        "--generation_mode",
        choices=["content", "entry"],
        default=GENERATION_MODE_CONTENT,
        help=(
            "`content` 不预插 payload 起始，允许模型直接续写 reasoning / 字面量 <acml:action>；"
            "`entry` 让模型从完整 me entry 起点开始写。"
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument(
        "--inference_device_map",
        choices=[INFERENCE_DEVICE_MAP_SINGLE, INFERENCE_DEVICE_MAP_AUTO, INFERENCE_DEVICE_MAP_CPU],
        default=INFERENCE_DEVICE_MAP_SINGLE,
        help=(
            "`single` 默认把模型放到单张 GPU，避免 trainable token adapter 被 auto 分片；"
            "`auto` 使用 Transformers/Accelerate 自动分片；`cpu` 强制 CPU。"
        ),
    )
    add_optional_bool_arg(
        parser,
        "--debug_tokens",
        default=False,
        help="打印原始生成 token id、逐 token 解码和结构 token 命中情况，用于诊断 ACML 结构输出。",
    )
    parser.add_argument(
        "--debug_token_limit",
        type=int,
        default=256,
        help="debug token 行和 id 列表最多展示多少个 token；设为 0 表示不截断。",
    )
    add_optional_bool_arg(parser, "--load_in_4bit", default=False)
    args = parser.parse_args()

    if args.interactive is None:
        args.interactive = args.prompt is None
    if not args.interactive and not args.prompt:
        parser.error("--interactive false 时必须提供 --prompt")
    return args


def build_inference_prompt_config(args: argparse.Namespace) -> InferencePromptConfig:
    return InferencePromptConfig(
        belief_prompt=args.belief_prompt,
        developer_name=args.developer_name,
        developer_entity_id=args.developer_entity_id,
        message_source=args.message_source,
        reply_tool_name=args.reply_tool_name,
    )


def generate_once(
    args: argparse.Namespace,
    encoder: AgenticContextEncoder,
    tokenizer,
    model,
    user_prompt: str,
) -> SingleTurnGenerationResult:
    return generate_single_turn_result(
        user_prompt,
        encoder=encoder,
        tokenizer=tokenizer,
        model=model,
        prompt_config=build_inference_prompt_config(args),
        generation_mode=args.generation_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def format_interactive_result(
    result: SingleTurnGenerationResult,
    *,
    args: argparse.Namespace,
    encoder: AgenticContextEncoder,
    tokenizer,
) -> str:
    rendered = format_generation_result(result)
    if not args.debug_tokens:
        return rendered
    return "\n".join(
        [
            rendered,
            format_generation_token_debug(
                result,
                encoder=encoder,
                tokenizer=tokenizer,
                max_tokens=args.debug_token_limit,
            ),
        ]
    )


def run_interactive(
    args: argparse.Namespace,
    encoder: AgenticContextEncoder,
    tokenizer,
    model,
    initial_prompt: str | None = None,
) -> None:
    print(
        "进入交互式单轮推理："
        f"kind=me, generation_mode={args.generation_mode}, developer={args.developer_name}, tool={args.reply_tool_name}, "
        f"temperature={args.temperature}, top_p={args.top_p}"
    )
    print("每次输入都会独立推理，不保留历史。输入 exit、quit 或 :q 退出。")

    if initial_prompt:
        print("\nobservation>")
        print(initial_prompt)
        print("\nme>")
        result = generate_once(args, encoder, tokenizer, model, initial_prompt)
        print(format_interactive_result(result, args=args, encoder=encoder, tokenizer=tokenizer))

    while True:
        try:
            user_prompt = input("\nobservation> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互。")
            return

        if not user_prompt:
            continue
        if user_prompt.lower() in EXIT_COMMANDS:
            print("退出交互。")
            return

        print("\nme>")
        result = generate_once(args, encoder, tokenizer, model, user_prompt)
        print(format_interactive_result(result, args=args, encoder=encoder, tokenizer=tokenizer))


def main() -> None:
    args = parse_args()
    tokenizer, model = load_lora_inference_model(
        args.model_name_or_path,
        args.adapter_path,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
        inference_device_map=args.inference_device_map,
    )
    encoder = AgenticContextEncoder(tokenizer)

    if args.interactive:
        run_interactive(args, encoder, tokenizer, model, initial_prompt=args.prompt)
        return

    result = generate_once(args, encoder, tokenizer, model, args.prompt or "")
    print(format_interactive_result(result, args=args, encoder=encoder, tokenizer=tokenizer))


if __name__ == "__main__":
    main()
