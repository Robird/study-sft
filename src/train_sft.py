"""LoRA/QLoRA SFT training script for the study-sft lab.

The default path is intentionally close to the already-verified Unsloth setup in
/repos/study-base-llm, with one extra axis: prompt_mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError as exc:
    raise SystemExit("Unsloth 未安装。请先在当前环境安装 unsloth。") from exc

import torch
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from study_sft.formats import (
    DEFAULT_BORA_REASONING,
    DEFAULT_SYSTEM_PROMPT,
    DatasetFormat,
    PromptMode,
    format_sft_text,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

LOGGER = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "/mnt/fast/LLM/Qwen3-1.7B-Base"
    local_files_only: bool = True
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_split: str = "train"
    dataset_streaming: bool = False
    dataset_format: DatasetFormat = "alpaca"
    prompt_mode: PromptMode = "chatml"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    bora_reasoning: str = DEFAULT_BORA_REASONING
    limit_train_samples: Optional[int] = None

    output_dir: str = "/mnt/fast/LLM/study-sft/qwen3-1.7b-chatml-lora"
    overwrite_output_dir: bool = False
    seed: int = 42

    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    optim: str = "adamw_torch"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False

    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = None
    bias: str = "none"
    load_in_4bit: bool = False
    max_length: int = 2048
    packing: bool = False
    dataloader_num_workers: int = 2
    ddp_find_unused_parameters: bool = False
    report_to: str = "none"
    lora_merge: bool = False


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "1", "y"}:
        return True
    if lowered in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=ScriptArguments.model_name_or_path)
    parser.add_argument("--local_files_only", nargs="?", const=True, type=str2bool, default=True)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config")
    parser.add_argument("--dataset_path")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--dataset_streaming", type=str2bool, default=False)
    parser.add_argument("--dataset_format", choices=["alpaca", "messages", "sharegpt", "text"], default="alpaca")
    parser.add_argument("--prompt_mode", choices=["chatml", "late_system", "bora"], default="chatml")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--bora_reasoning", default=DEFAULT_BORA_REASONING)
    parser.add_argument("--limit_train_samples", type=int)

    parser.add_argument("--output_dir", default=ScriptArguments.output_dir)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules")
    parser.add_argument("--bias", default="none")
    parser.add_argument("--load_in_4bit", nargs="?", const=True, type=str2bool, default=False)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--lora_merge", action="store_true")
    return ScriptArguments(**vars(parser.parse_args()))


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_model_and_tokenizer(args: ScriptArguments):
    LOGGER.info("使用 Unsloth 加载模型: %s", args.model_name_or_path)
    target_modules = (
        [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        if args.lora_target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_checkpointing = "unsloth" if args.gradient_checkpointing and world_size <= 1 else args.gradient_checkpointing
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=gradient_checkpointing,
        local_files_only=args.local_files_only,
        disable_log_stats=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.bias,
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()
    return model, tokenizer


def load_any_dataset(args: ScriptArguments):
    if args.dataset_path:
        path = Path(args.dataset_path)
        LOGGER.info("从本地路径加载数据集: %s", path)
        if path.is_file():
            return load_dataset("json", data_files=str(path), split=args.dataset_split)
        return load_from_disk(str(path))
    if not args.dataset_name:
        raise ValueError("必须指定 --dataset_path 或 --dataset_name")
    LOGGER.info("从 Hub 加载数据集: %s", args.dataset_name)
    return load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
    )


def format_dataset(args: ScriptArguments, dataset):
    def add_text(record: dict) -> dict[str, str]:
        return {
            "text": format_sft_text(
                record,
                dataset_format=args.dataset_format,
                prompt_mode=args.prompt_mode,
                default_system=args.system_prompt,
                bora_reasoning=args.bora_reasoning,
            )
        }

    if isinstance(dataset, IterableDataset):
        mapped = dataset.map(add_text)
        if args.limit_train_samples:
            mapped = mapped.take(args.limit_train_samples)
        return mapped

    if args.limit_train_samples:
        dataset = dataset.select(range(min(args.limit_train_samples, len(dataset))))
    remove_columns = list(dataset.column_names)
    return dataset.map(add_text, remove_columns=remove_columns, desc="format SFT text")


def build_trainer(args: ScriptArguments, model, tokenizer, train_dataset: Dataset | IterableDataset) -> SFTTrainer:
    report_to = "none" if args.report_to.lower() == "none" else args.report_to
    bf16_enabled = args.bf16 and is_bfloat16_supported()
    fp16_enabled = args.fp16 and not bf16_enabled

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        report_to=report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        max_length=args.max_length,
        packing=args.packing,
        dataset_text_field="text",
        seed=args.seed,
    )
    if training_args.eos_token in (None, "<EOS_TOKEN>"):
        training_args.eos_token = tokenizer.eos_token
    if training_args.pad_token in (None, "<PAD_TOKEN>"):
        training_args.pad_token = tokenizer.pad_token or tokenizer.eos_token

    return SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )


def get_last_checkpoint(output_dir: str) -> str | None:
    output = Path(output_dir)
    if not output.is_dir():
        return None
    checkpoints = [path for path in output.iterdir() if path.name.startswith("checkpoint-") and path.is_dir()]
    if not checkpoints:
        return None

    def step(path: Path) -> int:
        try:
            return int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            return -1

    checkpoint = max(checkpoints, key=step)
    return str(checkpoint) if (checkpoint / "trainer_state.json").exists() else None


def maybe_merge_and_save(trainer: SFTTrainer, args: ScriptArguments) -> None:
    if not args.lora_merge:
        return
    merged_dir = Path(args.output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained_merged(
        str(merged_dir),
        trainer.tokenizer,
        save_method="merged_16bit",
    )
    LOGGER.info("已保存合并模型: %s", merged_dir)


def main() -> None:
    setup_logging()
    args = parse_args()
    LOGGER.info("参数配置:\n%s", json.dumps(asdict(args), ensure_ascii=False, indent=2))
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    raw_dataset = load_any_dataset(args)
    train_dataset = format_dataset(args, raw_dataset)
    trainer = build_trainer(args, model, tokenizer, train_dataset)

    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        LOGGER.info("检测到 checkpoint，将恢复训练: %s", last_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    maybe_merge_and_save(trainer, args)
    LOGGER.info("训练完成: %s", args.output_dir)


if __name__ == "__main__":
    main()
