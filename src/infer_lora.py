"""Run one-off or interactive single-turn generation with a trained LoRA adapter."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from study_sft.formats import DEFAULT_SYSTEM_PROMPT, format_generation_prompt


EXIT_COMMANDS = {"exit", "quit", ":q"}


if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import AutoTokenizer


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "1", "y"}:
        return True
    if lowered in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="/mnt/fast/LLM/Qwen3-1.7B-Base")
    parser.add_argument("--local_files_only", nargs="?", const=True, type=str2bool, default=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--prompt")
    parser.add_argument(
        "--interactive",
        nargs="?",
        const=True,
        type=str2bool,
        default=None,
        help="进入交互式单轮推理；默认在未提供 --prompt 时自动开启",
    )
    parser.add_argument("--prompt_mode", choices=["chatml", "late_system", "bora"], default="chatml")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", nargs="?", const=True, type=str2bool, default=False)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    if args.interactive is None:
        args.interactive = args.prompt is None
    if not args.interactive and not args.prompt:
        parser.error("--interactive false 时必须提供 --prompt")
    return args


def load_tokenizer_and_model(args: argparse.Namespace) -> tuple[AutoTokenizer, PeftModel]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "dtype": dtype,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return tokenizer, model


def generate_once(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: PeftModel,
    user_prompt: str,
) -> str:
    import torch

    prompt = format_generation_prompt(
        user=user_prompt,
        prompt_mode=args.prompt_mode,
        system=args.system_prompt,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False).strip()


def run_interactive(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: PeftModel,
    initial_prompt: str | None = None,
) -> None:
    print(
        f"进入交互式单轮推理：prompt_mode={args.prompt_mode}, temperature={args.temperature}, top_p={args.top_p}"
    )
    print("每次输入都会独立推理，不保留历史。输入 exit、quit 或 :q 退出。")

    if initial_prompt:
        print("\nuser>")
        print(initial_prompt)
        print("\nassistant>")
        print(generate_once(args, tokenizer, model, initial_prompt) or "(空输出)")

    while True:
        try:
            user_prompt = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互。")
            return

        if not user_prompt:
            continue
        if user_prompt.lower() in EXIT_COMMANDS:
            print("退出交互。")
            return

        print("\nassistant>")
        print(generate_once(args, tokenizer, model, user_prompt) or "(空输出)")


def main() -> None:
    args = parse_args()
    tokenizer, model = load_tokenizer_and_model(args)

    if args.interactive:
        run_interactive(args, tokenizer, model, initial_prompt=args.prompt)
        return

    print(generate_once(args, tokenizer, model, args.prompt or ""))


if __name__ == "__main__":
    main()
