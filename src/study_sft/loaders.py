"""Shared loading helpers for datasets and tokenizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


DEFAULT_MODEL_NAME_OR_PATH = "/mnt/fast/LLM/Qwen3-1.7B-Base"


def ensure_tokenizer_pad_token(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_effective_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return eos_token_id


def load_base_tokenizer(model_name_or_path: str, *, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    return ensure_tokenizer_pad_token(tokenizer)


def load_dataset_source(
    *,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    logger: logging.Logger | None = None,
) -> Dataset:
    if dataset_path:
        path = Path(dataset_path)
        if logger is not None:
            logger.info("从本地路径加载数据集: %s", path)
        if path.is_file():
            return load_dataset("json", data_files=str(path), split=dataset_split)
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            if dataset_split not in loaded:
                raise ValueError(f"本地数据集目录缺少 split {dataset_split!r}: {path}")
            return loaded[dataset_split]
        return loaded

    if not dataset_name:
        raise ValueError("必须指定 --dataset_path 或 --dataset_name")
    if logger is not None:
        logger.info("从 Hub 加载数据集: %s", dataset_name)
    return load_dataset(dataset_name, dataset_config, split=dataset_split)
