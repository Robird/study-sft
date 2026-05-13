"""Dataset-level builders for pretokenized ACML SFT."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from datasets import Dataset

from study_sft.agentic_context import ENCODING_VERSION, AgenticContextEncoder, EncodedContext
from study_sft.training_cache import TrainingDatasetCacheStore
from study_sft.training_data import TrainingEncodingConfig, encode_training_features_from_record


TRAINING_DATASET_CACHE_VERSION = "agentic-training-dataset-v5"
ACML_DATA_PROTOCOL = "acml"


@dataclass(frozen=True)
class DatasetLocator:
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"


@dataclass(frozen=True)
class TrainingDatasetBuildOptions:
    validate_encoding: bool = False
    limit_train_samples: int | None = None
    cache_dir: Path | None = None


def parse_bloom_level_sampling_weights(spec: str | None) -> dict[str, float]:
    if spec is None:
        return {}
    items = [item.strip() for item in spec.split(",") if item.strip()]
    if not items:
        raise ValueError("bloom_level_sampling_weights 不能为空字符串")

    weights: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                "bloom_level_sampling_weights 必须使用 level=weight 形式，例如 remember=8,understand=2"
            )
        level, raw_weight = item.split("=", 1)
        bloom_level = level.strip()
        if not bloom_level:
            raise ValueError("bloom_level_sampling_weights 中存在空 bloom_level 名称")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"无法解析 bloom_level 权重: {item!r}") from exc
        if weight < 0:
            raise ValueError(f"bloom_level 权重不能为负数: {item!r}")
        weights[bloom_level] = weight
    return weights


def bloom_level_counts(dataset: Dataset) -> dict[str, int]:
    if "bloom_level" not in dataset.column_names:
        return {}
    counts: dict[str, int] = {}
    for index, value in enumerate(dataset["bloom_level"]):
        if not isinstance(value, str) or not value:
            raise ValueError(f"record[{index}] 缺少有效的 bloom_level 字段")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def resample_dataset_by_bloom_level(
    dataset: Dataset,
    *,
    weights: Mapping[str, float],
    seed: int,
    logger: logging.Logger | None = None,
) -> Dataset:
    if not weights or len(dataset) == 0:
        return dataset
    if "bloom_level" not in dataset.column_names:
        raise ValueError("当前数据集没有 bloom_level 列，无法应用 --bloom_level_sampling_weights")

    item_weights: list[float] = []
    for index, bloom_level in enumerate(dataset["bloom_level"]):
        if not isinstance(bloom_level, str) or not bloom_level:
            raise ValueError(f"record[{index}] 缺少有效的 bloom_level 字段")
        item_weights.append(weights.get(bloom_level, 1.0))

    if sum(item_weights) <= 0:
        raise ValueError("bloom_level_sampling_weights 使所有样本权重都变成了 0")

    if logger is not None:
        logger.info(
            "应用 bloom_level 重采样: weights=%s, before=%s",
            json.dumps(dict(sorted(weights.items())), ensure_ascii=False),
            json.dumps(bloom_level_counts(dataset), ensure_ascii=False),
        )

    sampled_indices = random.Random(seed).choices(range(len(dataset)), weights=item_weights, k=len(dataset))
    sampled_dataset = dataset.select(sampled_indices)

    if logger is not None:
        logger.info(
            "bloom_level 重采样完成: after=%s",
            json.dumps(bloom_level_counts(sampled_dataset), ensure_ascii=False),
        )
    return sampled_dataset


def tokenizer_identity_payload(encoder: AgenticContextEncoder) -> dict[str, object]:
    tokenizer = encoder.tokenizer
    token_table = encoder.policy.token_table
    probe_texts = (
        token_table.message_start_text,
        token_table.message_end_text,
        token_table.opaque_payload_start_text,
        token_table.opaque_payload_end_text,
        *[f"{kind}\n" for kind in encoder.policy.allowed_kinds],
        "\n",
        "agentic-cache-probe",
        "agentic-cache-probe <payload>",
    )
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "probe_encodings": {
            text: tokenizer.encode(text, add_special_tokens=False)
            for text in probe_texts
        },
    }


def _require_acml_column(dataset: Dataset) -> None:
    if "acml" not in dataset.column_names:
        raise ValueError("training_dataset expects a dataset with an 'acml' column")


def build_training_dataset_cache_identity(
    dataset: Dataset,
    *,
    encoder: AgenticContextEncoder,
    encoding_config: TrainingEncodingConfig,
    dataset_locator: DatasetLocator | None = None,
) -> dict[str, object]:
    _require_acml_column(dataset)
    locator = dataset_locator or DatasetLocator()
    token_table = encoder.policy.token_table
    return {
        "cache_version": TRAINING_DATASET_CACHE_VERSION,
        "encoding_version": ENCODING_VERSION,
        "data_protocol": ACML_DATA_PROTOCOL,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_locator": {
            "dataset_path": str(Path(locator.dataset_path).resolve()) if locator.dataset_path else None,
            "dataset_name": locator.dataset_name,
            "dataset_config": locator.dataset_config,
            "dataset_split": locator.dataset_split,
        },
        "max_length": encoding_config.max_length,
        "label_policy": encoding_config.label_policy,
        "tokenizer": tokenizer_identity_payload(encoder),
        "policy": {
            "allowed_kinds": list(encoder.policy.allowed_kinds),
            "extra_reserved_ids": list(encoder.policy.extra_reserved_ids),
            "token_table": {
                "message_start": token_table.message_start,
                "message_end": token_table.message_end,
                "opaque_payload_start": token_table.opaque_payload_start,
                "opaque_payload_end": token_table.opaque_payload_end,
                "message_start_text": token_table.message_start_text,
                "message_end_text": token_table.message_end_text,
                "opaque_payload_start_text": token_table.opaque_payload_start_text,
                "opaque_payload_end_text": token_table.opaque_payload_end_text,
            },
        },
    }


def training_dataset_cache_key(cache_payload: dict[str, object]) -> str:
    serialized = json.dumps(cache_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _empty_encoded_dataset() -> Dataset:
    return Dataset.from_dict({"input_ids": [], "labels": []})


def _validate_limit_train_samples(limit_train_samples: int) -> None:
    if limit_train_samples < 0:
        raise ValueError("limit_train_samples must be non-negative")


def validate_cached_training_dataset(dataset: Dataset, encoder: AgenticContextEncoder) -> None:
    for row in dataset:
        input_ids = list(row["input_ids"])
        labels = list(row["labels"])
        if len(labels) != len(input_ids):
            raise ValueError("cached training row has mismatched input_ids and labels lengths")
        encoder.validate(EncodedContext(input_ids=input_ids, loss_mask=[0] * len(input_ids)))


def _feature_columns_from_encoded_rows(
    encoded_rows: list[dict[str, list[int]]],
) -> dict[str, list[list[int]]]:
    return {
        "input_ids": [row["input_ids"] for row in encoded_rows],
        "labels": [row["labels"] for row in encoded_rows],
    }


def encode_training_dataset(
    dataset: Dataset,
    *,
    encoder: AgenticContextEncoder,
    encoding_config: TrainingEncodingConfig,
    validate_encoding: bool = False,
) -> Dataset:
    _require_acml_column(dataset)

    def add_features(batch: dict[str, list[object]]) -> dict[str, list[list[int]]]:
        batch_size = len(next(iter(batch.values()), []))
        records = [
            {
                column_name: column_values[index]
                for column_name, column_values in batch.items()
            }
            for index in range(batch_size)
        ]
        encoded_rows = [
            encode_training_features_from_record(
                dict(record),
                encoder=encoder,
                config=encoding_config,
                validate_encoding=validate_encoding,
            )
            for record in records
        ]
        return _feature_columns_from_encoded_rows(encoded_rows)

    return dataset.map(
        add_features,
        batched=True,
        remove_columns=list(dataset.column_names),
        desc="encode acml training samples",
    )


def _encode_training_dataset_limited(
    dataset: Dataset,
    *,
    encoder: AgenticContextEncoder,
    encoding_config: TrainingEncodingConfig,
    validate_encoding: bool,
    limit_train_samples: int,
) -> Dataset:
    _require_acml_column(dataset)
    _validate_limit_train_samples(limit_train_samples)
    if limit_train_samples == 0:
        return _empty_encoded_dataset()
    encoded_rows: list[dict[str, list[int]]] = []
    for record in dataset:
        features = encode_training_features_from_record(
            dict(record),
            encoder=encoder,
            config=encoding_config,
            validate_encoding=validate_encoding,
        )
        encoded_rows.append(features)
        if len(encoded_rows) >= limit_train_samples:
            break
    return Dataset.from_dict(_feature_columns_from_encoded_rows(encoded_rows))


def prepare_training_dataset(
    dataset: Dataset,
    *,
    encoder: AgenticContextEncoder,
    encoding_config: TrainingEncodingConfig,
    build_options: TrainingDatasetBuildOptions | None = None,
    dataset_locator: DatasetLocator | None = None,
    logger: logging.Logger | None = None,
) -> Dataset:
    _require_acml_column(dataset)
    options = build_options or TrainingDatasetBuildOptions()
    locator = dataset_locator or DatasetLocator()
    validate_encoding = options.validate_encoding
    limit_train_samples = options.limit_train_samples
    if limit_train_samples == 0:
        return _empty_encoded_dataset()
    if limit_train_samples is not None:
        return _encode_training_dataset_limited(
            dataset,
            encoder=encoder,
            encoding_config=encoding_config,
            validate_encoding=validate_encoding,
            limit_train_samples=limit_train_samples,
        )
    if options.cache_dir is None:
        return encode_training_dataset(
            dataset,
            encoder=encoder,
            encoding_config=encoding_config,
            validate_encoding=validate_encoding,
        )

    cache_payload = build_training_dataset_cache_identity(
        dataset,
        encoder=encoder,
        encoding_config=encoding_config,
        dataset_locator=locator,
    )
    cache_key = training_dataset_cache_key(cache_payload)
    store = TrainingDatasetCacheStore(options.cache_dir, logger=logger)
    return store.load_or_store(
        cache_key,
        metadata=cache_payload,
        build=lambda: encode_training_dataset(
            dataset,
            encoder=encoder,
            encoding_config=encoding_config,
            validate_encoding=validate_encoding,
        ),
        validate=(
            (lambda cached: validate_cached_training_dataset(cached, encoder))
            if validate_encoding
            else None
        ),
    )
