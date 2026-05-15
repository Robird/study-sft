"""Small shared argparse helpers for repository scripts."""

from __future__ import annotations

import argparse

from study_sft.inference_prompts import (
    DEFAULT_BELIEF_PROMPT,
    DEFAULT_DEVELOPER_NAME,
    DEFAULT_MESSAGE_SOURCE,
    DEFAULT_REPLY_TOOL_NAME,
)
from study_sft.loaders import DEFAULT_MODEL_NAME_OR_PATH


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
    parser.add_argument(
        "--dataset_path",
        help=(
            "本地数据源路径。支持单个 .acml 文件、包含 acml 列的 JSON/JSONL 文件、"
            "按 bloom_level 分 shard 的根目录（递归读取 */data.jsonl），"
            "或 datasets.save_to_disk 导出的本地目录。"
        ),
    )
    parser.add_argument("--dataset_split", default="train", help="Hub dataset 或本地 datasets 目录的 split 名称")


def add_belief_prompt_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--belief_prompt",
        default=DEFAULT_BELIEF_PROMPT,
        help="追加到 belief entry 中的补充说明文本。",
    )


def add_inference_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--developer_name",
        default=DEFAULT_DEVELOPER_NAME,
        help="推理时写入 observation/belief 的开发者名称。",
    )
    parser.add_argument(
        "--developer_entity_id",
        default=None,
        help="可选：显式指定开发者 entity_id；默认按 developer_name 稳定生成。",
    )
    parser.add_argument(
        "--message_source",
        default=DEFAULT_MESSAGE_SOURCE,
        help="推理 observation 中使用的消息来源描述。",
    )
    parser.add_argument(
        "--reply_tool_name",
        default=DEFAULT_REPLY_TOOL_NAME,
        help="belief 中注入的对外消息工具原型名称。",
    )
