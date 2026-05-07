"""Run a quick generation test with a trained LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from study_sft.formats import DEFAULT_SYSTEM_PROMPT, format_generation_prompt


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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt_mode", choices=["chatml", "late_system", "bora"], default="chatml")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", nargs="?", const=True, type=str2bool, default=False)
    parser.add_argument("--max_length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    prompt = format_generation_prompt(
        user=args.prompt,
        prompt_mode=args.prompt_mode,
        system=args.system_prompt,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
