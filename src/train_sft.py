"""LoRA/QLoRA training script for ACML-based agentic-context SFT."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from study_sft.cli_args import (
    add_dataset_source_args,
    add_model_source_args,
    add_optional_bool_arg,
)
from study_sft.agentic_context import AgenticContextEncoder
from study_sft.loaders import (
    DEFAULT_MODEL_NAME_OR_PATH,
    ensure_tokenizer_pad_token,
    get_effective_pad_token_id,
    load_base_tokenizer,
    load_dataset_source,
)
from study_sft.training_data import TrainingEncodingConfig, TrainingLabelPolicy
from study_sft.training_dataset import (
    DatasetLocator,
    TrainingDatasetBuildOptions,
    bloom_level_counts,
    parse_bloom_level_sampling_weights,
    prepare_training_dataset,
    resample_dataset_by_bloom_level,
    tokenizer_identity_payload,
)
from study_sft.training_runtime import AgenticDataCollator


LOGGER = logging.getLogger(__name__)
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _require_unsloth():
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as exc:
        raise SystemExit("Unsloth 未安装。请先在当前环境安装 unsloth。") from exc
    return FastLanguageModel, is_bfloat16_supported


def configure_torch_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")


@dataclass
class ScriptArguments:
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    local_files_only: bool = True
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_split: str = "train"
    limit_train_samples: Optional[int] = None
    bloom_level_sampling_weights: Optional[str] = None
    label_policy: TrainingLabelPolicy = "entry"

    output_dir: str = "/mnt/fast/LLM/study-sft/qwen3-1.7b-agentic-lora"
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
    validate_encoding: bool = False
    dataloader_num_workers: int = 2
    ddp_find_unused_parameters: bool = False
    report_to: str = "none"
    lora_merge: bool = False
    cache_train_dataset: bool = True
    train_dataset_cache_dir: Optional[str] = ".cache/study_sft/train_datasets"


def parse_args() -> ScriptArguments:
    defaults = ScriptArguments()
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_source_args(parser, default_model_name=defaults.model_name_or_path)
    add_dataset_source_args(parser)
    parser.add_argument("--limit_train_samples", type=int)
    parser.add_argument(
        "--bloom_level_sampling_weights",
        default=defaults.bloom_level_sampling_weights,
        help=(
            "按 bloom_level 做训练前重采样，格式如 remember=8,understand=2,apply=1。"
            "未列出的 level 默认权重为 1；0 表示在该轮训练中排除。"
        ),
    )
    parser.add_argument("--label_policy", choices=["entry", "payload_only"], default=defaults.label_policy)

    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num_train_epochs", type=float, default=defaults.num_train_epochs)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--per_device_train_batch_size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--optim", default=defaults.optim)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--logging_steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--save_steps", type=int, default=defaults.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=defaults.save_total_limit)
    add_optional_bool_arg(parser, "--gradient_checkpointing", default=defaults.gradient_checkpointing)
    add_optional_bool_arg(parser, "--bf16", default=defaults.bf16)
    add_optional_bool_arg(parser, "--fp16", default=defaults.fp16)
    parser.add_argument("--lora_r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--lora_target_modules")
    parser.add_argument("--bias", default=defaults.bias)
    add_optional_bool_arg(parser, "--load_in_4bit", default=defaults.load_in_4bit)
    parser.add_argument("--max_length", type=int, default=defaults.max_length)
    add_optional_bool_arg(parser, "--validate_encoding", default=defaults.validate_encoding)
    parser.add_argument("--dataloader_num_workers", type=int, default=defaults.dataloader_num_workers)
    add_optional_bool_arg(parser, "--ddp_find_unused_parameters", default=defaults.ddp_find_unused_parameters)
    parser.add_argument("--report_to", default=defaults.report_to)
    parser.add_argument("--lora_merge", action="store_true")
    add_optional_bool_arg(parser, "--cache_train_dataset", default=defaults.cache_train_dataset)
    parser.add_argument("--train_dataset_cache_dir", default=defaults.train_dataset_cache_dir)
    return ScriptArguments(**vars(parser.parse_args()))


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def resolve_lora_target_modules(args: ScriptArguments) -> list[str]:
    if not args.lora_target_modules:
        return list(DEFAULT_LORA_TARGET_MODULES)
    return [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]


def resolve_gradient_checkpointing_mode(args: ScriptArguments) -> bool | str:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.gradient_checkpointing and world_size <= 1:
        return "unsloth"
    return args.gradient_checkpointing


def build_training_args(args: ScriptArguments) -> TrainingArguments:
    _, is_bfloat16_supported = _require_unsloth()
    report_to = "none" if args.report_to.lower() == "none" else args.report_to
    bf16_enabled = args.bf16 and is_bfloat16_supported()
    fp16_enabled = args.fp16 and not bf16_enabled
    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
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
        seed=args.seed,
        remove_unused_columns=False,
    )


def load_model_and_tokenizer(args: ScriptArguments):
    FastLanguageModel, _ = _require_unsloth()
    LOGGER.info("使用 Unsloth 加载模型: %s", args.model_name_or_path)
    target_modules = resolve_lora_target_modules(args)
    gradient_checkpointing = resolve_gradient_checkpointing_mode(args)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=gradient_checkpointing,
        local_files_only=args.local_files_only,
        disable_log_stats=True,
    )
    ensure_tokenizer_pad_token(tokenizer)

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


def assert_matching_training_tokenizer(
    encoded_dataset_encoder: AgenticContextEncoder,
    training_tokenizer,
) -> None:
    training_encoder = AgenticContextEncoder(training_tokenizer)
    if tokenizer_identity_payload(encoded_dataset_encoder) != tokenizer_identity_payload(training_encoder):
        raise ValueError("训练 tokenizer 与预编码 tokenizer 不一致，无法安全复用已编码数据")


def build_trainer(args: ScriptArguments, model, tokenizer, train_dataset: Dataset) -> Trainer:
    return Trainer(
        model=model,
        args=build_training_args(args),
        data_collator=AgenticDataCollator(get_effective_pad_token_id(tokenizer)),
        train_dataset=train_dataset,
        processing_class=tokenizer,
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


def resolve_resume_checkpoint(output_dir: str, *, overwrite_output_dir: bool) -> str | None:
    if overwrite_output_dir:
        return None
    output = Path(output_dir)
    if not output.exists() or not any(output.iterdir()):
        return None
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None:
        raise ValueError(f"输出目录已存在且非空，请使用新目录或传入 --overwrite_output_dir: {output_dir}")
    return last_checkpoint


def build_train_dataset(
    args: ScriptArguments,
    encoder: AgenticContextEncoder,
) -> Dataset:
    build_options = TrainingDatasetBuildOptions(
        validate_encoding=args.validate_encoding,
        limit_train_samples=args.limit_train_samples,
        cache_dir=(
            Path(args.train_dataset_cache_dir)
            if args.cache_train_dataset and args.train_dataset_cache_dir
            else None
        ),
    )
    dataset_locator = DatasetLocator(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
    )
    raw_dataset = load_dataset_source(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        logger=LOGGER,
    )
    sampling_weights = parse_bloom_level_sampling_weights(args.bloom_level_sampling_weights)
    if sampling_weights:
        raw_bloom_counts = bloom_level_counts(raw_dataset)
        if raw_bloom_counts:
            LOGGER.info("原始 bloom_level 分布: %s", json.dumps(raw_bloom_counts, ensure_ascii=False))
    raw_dataset = resample_dataset_by_bloom_level(
        raw_dataset,
        weights=sampling_weights,
        seed=args.seed,
        logger=LOGGER,
    )
    return prepare_training_dataset(
        raw_dataset,
        encoder=encoder,
        encoding_config=TrainingEncodingConfig(
            max_length=args.max_length,
            label_policy=args.label_policy,
        ),
        build_options=build_options,
        dataset_locator=dataset_locator,
        logger=LOGGER,
    )


def maybe_merge_and_save(trainer: Trainer, tokenizer, args: ScriptArguments) -> None:
    if not args.lora_merge:
        return
    merged_dir = Path(args.output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    LOGGER.info("已保存合并模型: %s", merged_dir)


def main() -> None:
    configure_torch_runtime()
    setup_logging()
    args = parse_args()
    LOGGER.info("参数配置:\n%s", json.dumps(asdict(args), ensure_ascii=False, indent=2))
    set_seed(args.seed)

    last_checkpoint = resolve_resume_checkpoint(
        args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
    )

    base_tokenizer = load_base_tokenizer(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    dataset_encoder = AgenticContextEncoder(base_tokenizer)
    train_dataset = build_train_dataset(args, dataset_encoder)
    if len(train_dataset) == 0:
        raise ValueError("编码后的训练集为空，请检查数据源或 --limit_train_samples")
    model, tokenizer = load_model_and_tokenizer(args)
    assert_matching_training_tokenizer(dataset_encoder, tokenizer)
    trainer = build_trainer(args, model, tokenizer, train_dataset)

    if last_checkpoint:
        LOGGER.info("检测到 checkpoint，将恢复训练: %s", last_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    maybe_merge_and_save(trainer, tokenizer, args)
    LOGGER.info("训练完成: %s", args.output_dir)


if __name__ == "__main__":
    main()
