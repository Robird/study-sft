"""Persistent cache backend for encoded training datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from datasets import Dataset, load_from_disk


_CACHE_METADATA_FILENAME = "cache_meta.json"


def training_dataset_cache_metadata_path(cache_path: Path) -> Path:
    return cache_path / _CACHE_METADATA_FILENAME


def is_complete_training_dataset_cache_path(cache_path: Path) -> bool:
    return cache_path.is_dir() and training_dataset_cache_metadata_path(cache_path).is_file()


def load_training_dataset_cache_metadata(cache_path: Path) -> dict[str, Any]:
    return json.loads(training_dataset_cache_metadata_path(cache_path).read_text(encoding="utf-8"))


def write_training_dataset_cache_metadata(cache_path: Path, metadata: dict[str, Any]) -> None:
    training_dataset_cache_metadata_path(cache_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def remove_training_dataset_cache_path(cache_path: Path) -> None:
    if cache_path.is_dir():
        shutil.rmtree(cache_path)
    elif cache_path.exists():
        cache_path.unlink()


class TrainingDatasetCacheStore:
    def __init__(
        self,
        cache_dir: Path,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.logger = logger

    def load_or_store(
        self,
        cache_key: str,
        *,
        metadata: dict[str, Any],
        build: Callable[[], Dataset],
        validate: Callable[[Dataset], None] | None = None,
    ) -> Dataset:
        cache_path = self.cache_dir / cache_key
        cached = self._load_cached_dataset(cache_path, validate=validate)
        if cached is not None:
            return cached
        return self._encode_and_store_dataset(
            cache_path,
            cache_key=cache_key,
            metadata=metadata,
            build=build,
            validated=validate is not None,
        )

    def _load_cached_dataset(
        self,
        cache_path: Path,
        *,
        validate: Callable[[Dataset], None] | None,
    ) -> Dataset | None:
        if is_complete_training_dataset_cache_path(cache_path):
            if self.logger is not None:
                self.logger.info("命中训练编码缓存: %s", cache_path)
            cached = load_from_disk(str(cache_path))
            metadata = load_training_dataset_cache_metadata(cache_path)
            if validate is not None and not metadata.get("validated", False):
                if self.logger is not None:
                    self.logger.info("缓存存在但尚未验证，正在校验: %s", cache_path)
                validate(cached)
                metadata["validated"] = True
                write_training_dataset_cache_metadata(cache_path, metadata)
            return cached
        if cache_path.exists():
            if self.logger is not None:
                self.logger.warning("发现不完整训练缓存，正在删除后重建: %s", cache_path)
            remove_training_dataset_cache_path(cache_path)
        return None

    def _encode_and_store_dataset(
        self,
        cache_path: Path,
        *,
        cache_key: str,
        metadata: dict[str, Any],
        build: Callable[[], Dataset],
        validated: bool,
    ) -> Dataset:
        if self.logger is not None:
            self.logger.info("未命中训练编码缓存，开始构建: %s", cache_path)
        encoded = build()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_cache_path: Path | None = Path(
            tempfile.mkdtemp(prefix=f".tmp-{cache_key}-", dir=str(self.cache_dir))
        )
        try:
            assert temp_cache_path is not None
            encoded.save_to_disk(str(temp_cache_path))
            write_training_dataset_cache_metadata(
                temp_cache_path,
                {
                    **metadata,
                    "cache_key": cache_key,
                    "validated": validated,
                },
            )
            temp_cache_path.rename(cache_path)
        except FileExistsError:
            assert temp_cache_path is not None
            if is_complete_training_dataset_cache_path(cache_path):
                remove_training_dataset_cache_path(temp_cache_path)
            else:
                if self.logger is not None:
                    self.logger.warning("缓存目标已存在但不完整，使用当前写入重建: %s", cache_path)
                remove_training_dataset_cache_path(cache_path)
                temp_cache_path.rename(cache_path)
            temp_cache_path = None
        else:
            temp_cache_path = None
        finally:
            if temp_cache_path is not None and temp_cache_path.exists():
                remove_training_dataset_cache_path(temp_cache_path)
        if self.logger is not None:
            self.logger.info("已写入训练编码缓存: %s", cache_path)
        return load_from_disk(str(cache_path))
