"""Small shared argparse helpers for repository scripts."""

from __future__ import annotations

import argparse

from study_sft.loaders import DEFAULT_MODEL_NAME_OR_PATH
from study_sft.samples import DATASET_FORMAT_CHOICES, DEFAULT_BELIEF_PROMPT


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "1", "y"}:
        return True
    if lowered in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def add_optional_bool_arg(
    parser: argparse.ArgumentParser,
    *name_or_flags: str,
    default: bool | None,
    help: str | None = None,
) -> None:
    parser.add_argument(
        *name_or_flags,
        nargs="?",
        const=True,
        type=str2bool,
        default=default,
        help=help,
    )


def add_model_source_args(
    parser: argparse.ArgumentParser,
    *,
    default_model_name: str = DEFAULT_MODEL_NAME_OR_PATH,
) -> None:
    parser.add_argument("--model_name_or_path", default=default_model_name)
    add_optional_bool_arg(parser, "--local_files_only", default=True)


def add_dataset_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config")
    parser.add_argument("--dataset_path")
    parser.add_argument("--dataset_split", default="train")


def add_dataset_format_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_format", choices=DATASET_FORMAT_CHOICES, default="alpaca")


def add_belief_prompt_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--belief_prompt", default=DEFAULT_BELIEF_PROMPT)
