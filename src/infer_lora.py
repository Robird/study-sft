"""Run one-off or interactive single-turn generation with a trained LoRA adapter."""

from __future__ import annotations

import argparse

from study_sft.cli_args import add_belief_prompt_arg, add_model_source_args, add_optional_bool_arg
from study_sft.agentic_context import AgenticContextEncoder
from study_sft.inference_runtime import (
    SingleTurnGenerationResult,
    format_generation_result,
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
    add_belief_prompt_arg(parser)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    add_optional_bool_arg(parser, "--load_in_4bit", default=False)
    args = parser.parse_args()

    if args.interactive is None:
        args.interactive = args.prompt is None
    if not args.interactive and not args.prompt:
        parser.error("--interactive false 时必须提供 --prompt")
    return args


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
        belief_prompt=args.belief_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def run_interactive(
    args: argparse.Namespace,
    encoder: AgenticContextEncoder,
    tokenizer,
    model,
    initial_prompt: str | None = None,
) -> None:
    print(
        f"进入交互式单轮推理：role=me, temperature={args.temperature}, top_p={args.top_p}"
    )
    print("每次输入都会独立推理，不保留历史。输入 exit、quit 或 :q 退出。")

    if initial_prompt:
        print("\nobservation>")
        print(initial_prompt)
        print("\nme>")
        print(format_generation_result(generate_once(args, encoder, tokenizer, model, initial_prompt)))

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
        print(format_generation_result(generate_once(args, encoder, tokenizer, model, user_prompt)))


def main() -> None:
    args = parse_args()
    tokenizer, model = load_lora_inference_model(
        args.model_name_or_path,
        args.adapter_path,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
    )
    encoder = AgenticContextEncoder(tokenizer)

    if args.interactive:
        run_interactive(args, encoder, tokenizer, model, initial_prompt=args.prompt)
        return

    print(format_generation_result(generate_once(args, encoder, tokenizer, model, args.prompt or "")))


if __name__ == "__main__":
    main()
