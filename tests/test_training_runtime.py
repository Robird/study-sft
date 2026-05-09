from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset

import study_sft.training_cache as training_cache
import study_sft.training_dataset as training_dataset
from study_sft.agentic_context import AgenticContextEncoder, QWEN3_AGENTIC_TOKEN_TABLE
from study_sft.training_data import TrainingEncodingConfig
from study_sft.training_dataset import (
    DatasetLocator,
    TrainingDatasetBuildOptions,
    build_training_dataset_cache_identity,
    encode_training_dataset,
    prepare_training_dataset,
    tokenizer_identity_payload,
    training_dataset_cache_key,
    validate_cached_training_dataset,
)
from study_sft.training_runtime import AgenticDataCollator

from tests.test_training_data import FakeTokenizer


class ShiftedTextTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        input_ids = super().encode(text, add_special_tokens=add_special_tokens)
        reserved = set(self.special_ids.values())
        return [token_id if token_id in reserved else token_id + 10000 for token_id in input_ids]


def make_encoding_config(
    *,
    dataset_format: str = "alpaca",
    default_belief_prompt: str = "You are a tester.",
    max_length: int = 128,
    label_policy: str = "message",
) -> TrainingEncodingConfig:
    return TrainingEncodingConfig(
        dataset_format=dataset_format,
        default_belief_prompt=default_belief_prompt,
        max_length=max_length,
        label_policy=label_policy,
    )


def make_build_options(
    *,
    validate_encoding: bool = False,
    limit_train_samples: int | None = None,
    cache_dir: Path | None = None,
) -> TrainingDatasetBuildOptions:
    return TrainingDatasetBuildOptions(
        validate_encoding=validate_encoding,
        limit_train_samples=limit_train_samples,
        cache_dir=cache_dir,
    )


def make_dataset_locator(
    *,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str = "train",
) -> DatasetLocator:
    return DatasetLocator(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
    )


class TrainingRuntimeTests(unittest.TestCase):
    def test_encode_training_dataset_removes_source_columns(self) -> None:
        dataset = Dataset.from_dict(
            {
                "instruction": ["Explain SFT", "Explain LoRA"],
                "output": ["Answer one", "Answer two"],
            }
        )

        encoded = encode_training_dataset(
            dataset,
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(),
        )

        self.assertEqual(len(encoded), 2)
        self.assertEqual(encoded.column_names, ["input_ids", "labels"])

    def test_prepare_training_dataset_allows_explicit_zero_limit(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})

        encoded = prepare_training_dataset(
            dataset,
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(),
            build_options=make_build_options(limit_train_samples=0),
        )

        self.assertEqual(len(encoded), 0)

    def test_prepare_training_dataset_rejects_negative_limit(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            prepare_training_dataset(
                dataset,
                encoder=AgenticContextEncoder(FakeTokenizer()),
                encoding_config=make_encoding_config(),
                build_options=make_build_options(limit_train_samples=-1),
            )

    def test_prepare_training_dataset_applies_limit_after_expansion(self) -> None:
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "Question 1"},
                        {"role": "assistant", "content": "Answer 1"},
                        {"role": "user", "content": "Question 2"},
                        {"role": "assistant", "content": "Answer 2"},
                    ]
                ]
            }
        )

        encoded = prepare_training_dataset(
            dataset,
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(dataset_format="messages"),
            build_options=make_build_options(
                limit_train_samples=1,
                cache_dir=Path(tempfile.mkdtemp()),
            ),
        )

        self.assertEqual(len(encoded), 1)

    def test_prepare_training_dataset_reuses_cached_encoded_dataset(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})
        encoder = AgenticContextEncoder(FakeTokenizer())

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            first = prepare_training_dataset(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(),
                build_options=make_build_options(cache_dir=cache_dir),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )

            with patch(
                "study_sft.training_dataset.encode_training_dataset",
                side_effect=AssertionError("cache hit should not re-encode"),
            ):
                second = prepare_training_dataset(
                    dataset,
                    encoder=encoder,
                    encoding_config=make_encoding_config(),
                    build_options=make_build_options(cache_dir=cache_dir),
                    dataset_locator=make_dataset_locator(dataset_name="unit-test"),
                )

        self.assertEqual(first[0]["input_ids"], second[0]["input_ids"])

    def test_prepare_training_dataset_validates_uncertified_cache_on_hit(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})
        encoder = AgenticContextEncoder(FakeTokenizer())

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            prepare_training_dataset(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(),
                build_options=make_build_options(
                    validate_encoding=False,
                    cache_dir=cache_dir,
                ),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )

            cache_payload = build_training_dataset_cache_identity(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )
            cache_key = training_dataset_cache_key(cache_payload)
            metadata_path = cache_dir / cache_key / "cache_meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["validated"])

            with patch(
                "study_sft.training_dataset.encode_training_dataset",
                side_effect=AssertionError("cache hit should not re-encode"),
            ):
                second = prepare_training_dataset(
                    dataset,
                    encoder=encoder,
                    encoding_config=make_encoding_config(),
                    build_options=make_build_options(
                        validate_encoding=True,
                        cache_dir=cache_dir,
                    ),
                    dataset_locator=make_dataset_locator(dataset_name="unit-test"),
                )

            self.assertEqual(len(second), 1)
            updated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(updated_metadata["validated"])

    def test_validate_cached_training_dataset_rejects_mismatched_label_lengths(self) -> None:
        cached_dataset = Dataset.from_dict(
            {
                "input_ids": [[QWEN3_AGENTIC_TOKEN_TABLE.message_start]],
                "labels": [[]],
            }
        )

        with self.assertRaisesRegex(ValueError, "mismatched input_ids and labels lengths"):
            validate_cached_training_dataset(cached_dataset, AgenticContextEncoder(FakeTokenizer()))

    def test_prepare_training_dataset_rebuilds_cache_when_tokenizer_identity_changes(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            prepare_training_dataset(
                dataset,
                encoder=AgenticContextEncoder(FakeTokenizer()),
                encoding_config=make_encoding_config(),
                build_options=make_build_options(cache_dir=cache_dir),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )
            cache_dir_entries = sorted(path.name for path in cache_dir.iterdir() if path.is_dir())

            prepare_training_dataset(
                dataset,
                encoder=AgenticContextEncoder(ShiftedTextTokenizer()),
                encoding_config=make_encoding_config(),
                build_options=make_build_options(cache_dir=cache_dir),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )
            updated_cache_dir_entries = sorted(path.name for path in cache_dir.iterdir() if path.is_dir())

        self.assertEqual(len(cache_dir_entries), 1)
        self.assertEqual(len(updated_cache_dir_entries), 2)

    def test_cache_key_changes_when_label_policy_changes(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})
        encoder = AgenticContextEncoder(FakeTokenizer())

        message_key = training_dataset_cache_key(
            build_training_dataset_cache_identity(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(label_policy="message"),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )
        )
        payload_key = training_dataset_cache_key(
            build_training_dataset_cache_identity(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(label_policy="payload_only"),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )
        )

        self.assertNotEqual(message_key, payload_key)

    def test_prepare_training_dataset_ignores_incomplete_cache_dir(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})
        encoder = AgenticContextEncoder(FakeTokenizer())

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_payload = build_training_dataset_cache_identity(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(),
                dataset_locator=make_dataset_locator(dataset_name="unit-test"),
            )
            cache_path = cache_dir / training_dataset_cache_key(cache_payload)
            cache_path.mkdir(parents=True)

            with patch(
                "study_sft.training_dataset.encode_training_dataset",
                wraps=training_dataset.encode_training_dataset,
            ) as encode_mock:
                encoded = prepare_training_dataset(
                    dataset,
                    encoder=encoder,
                    encoding_config=make_encoding_config(),
                    build_options=make_build_options(cache_dir=cache_dir),
                    dataset_locator=make_dataset_locator(dataset_name="unit-test"),
                )

        self.assertEqual(len(encoded), 1)
        self.assertEqual(encode_mock.call_count, 1)

    def test_prepare_training_dataset_recovers_after_partial_cache_write_failure(self) -> None:
        dataset = Dataset.from_dict({"instruction": ["Explain SFT"], "output": ["Answer one"]})
        encoder = AgenticContextEncoder(FakeTokenizer())

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            with patch(
                "study_sft.training_cache.write_training_dataset_cache_metadata",
                side_effect=RuntimeError("simulated metadata write failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated metadata write failure"):
                    prepare_training_dataset(
                        dataset,
                        encoder=encoder,
                        encoding_config=make_encoding_config(),
                        build_options=make_build_options(cache_dir=cache_dir),
                        dataset_locator=make_dataset_locator(dataset_name="unit-test"),
                    )

            self.assertEqual([path.name for path in cache_dir.iterdir()], [])

    def test_agentic_data_collator_pads_features_consistently(self) -> None:
        collator = AgenticDataCollator(pad_token_id=QWEN3_AGENTIC_TOKEN_TABLE.message_end)
        batch = collator(
            [
                {"input_ids": [1, 2], "labels": [-100, 2]},
                {"input_ids": [3], "labels": [3]},
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, QWEN3_AGENTIC_TOKEN_TABLE.message_end]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])


if __name__ == "__main__":
    unittest.main()
