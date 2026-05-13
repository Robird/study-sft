"""Shared loading helpers for datasets and tokenizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


DEFAULT_MODEL_NAME_OR_PATH = "/mnt/fast/LLM/Qwen3-1.7B-Base"
DEFAULT_PAD_TOKEN_TEXT = "<|PAD_TOKEN|>"


def _normalize_acml_dataset_columns(dataset: Dataset) -> Dataset:
    if "acml" in dataset.column_names or "text" not in dataset.column_names:
        return dataset
    if len(dataset) == 0:
        return dataset
    sample = dataset[0].get("text")
    if isinstance(sample, str) and sample.lstrip().startswith("<acml"):
        return dataset.rename_column("text", "acml")
    return dataset


def _require_acml_column(dataset: Dataset) -> Dataset:
    dataset = _normalize_acml_dataset_columns(dataset)
    if "acml" not in dataset.column_names:
        raise ValueError("ACML dataset source must contain a string column named 'acml'")
    return dataset


def _load_local_json_dataset(data_files: str | list[str], *, dataset_split: str) -> Dataset:
    dataset = load_dataset("json", data_files=data_files, split=dataset_split)
    return _require_acml_column(dataset)


def _discover_shard_jsonl_files(path: Path) -> list[str]:
    return sorted(str(child) for child in path.rglob("data.jsonl") if child.is_file())


def ensure_tokenizer_pad_token(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token", None) is not None:
        return tokenizer
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        pad_token_id = convert_tokens_to_ids(DEFAULT_PAD_TOKEN_TEXT)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(pad_token_id, int) and pad_token_id >= 0 and pad_token_id != unk_token_id:
            tokenizer.pad_token = DEFAULT_PAD_TOKEN_TEXT
            return tokenizer
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
            if path.suffix == ".acml":
                return Dataset.from_dict({"acml": [path.read_text(encoding="utf-8")]})
            return _load_local_json_dataset(str(path), dataset_split=dataset_split)
        shard_jsonl_files = _discover_shard_jsonl_files(path)
        if shard_jsonl_files:
            if logger is not None:
                logger.info("从本地 shard 目录加载 JSONL 数据: %s (shards=%d)", path, len(shard_jsonl_files))
            return _load_local_json_dataset(shard_jsonl_files, dataset_split=dataset_split)
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            if dataset_split not in loaded:
                raise ValueError(f"本地数据集目录缺少 split {dataset_split!r}: {path}")
            return _require_acml_column(loaded[dataset_split])
        return _require_acml_column(loaded)

    if not dataset_name:
        raise ValueError("必须指定 --dataset_path 或 --dataset_name")
    if logger is not None:
        logger.info("从 Hub 加载数据集: %s", dataset_name)
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    return _require_acml_column(dataset)
