from __future__ import annotations

import json
from pathlib import Path
import tempfile
import train_sft
import unittest
from unittest.mock import patch

from datasets import Dataset

import study_sft.training_cache as training_cache
import study_sft.training_dataset as training_dataset
from study_sft.agentic_context import AgenticContextEncoder, QWEN3_AGENTIC_TOKEN_TABLE
from study_sft.loaders import load_dataset_source
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
    max_length: int = 128,
    label_policy: str = "entry",
) -> TrainingEncodingConfig:
    return TrainingEncodingConfig(
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


def make_acml_document(
    *,
    observation: str = "Explain SFT",
    answer: str = "Answer one",
    belief: str | None = None,
) -> str:
    belief_entry = ""
    if belief is not None:
        belief_entry = f'<acml:entry kind="belief">{belief}</acml:entry>'
    return (
        '<acml version="0">'
        f"{belief_entry}"
        f'<acml:entry kind="observation">{observation}</acml:entry>'
        f'<acml:entry kind="me" loss="true">{answer}</acml:entry>'
        "</acml>"
    )


def make_acml_dataset(*documents: str) -> Dataset:
    return Dataset.from_dict({"acml": list(documents)})


class TrainingRuntimeTests(unittest.TestCase):
    def test_parse_bloom_level_sampling_weights_parses_mapping(self) -> None:
        self.assertEqual(
            training_dataset.parse_bloom_level_sampling_weights("remember=8, understand=2,apply=1"),
            {"remember": 8.0, "understand": 2.0, "apply": 1.0},
        )

    def test_resample_dataset_by_bloom_level_can_exclude_a_bucket(self) -> None:
        dataset = Dataset.from_dict(
            {
                "acml": [
                    make_acml_document(observation="Q1", answer="A1"),
                    make_acml_document(observation="Q2", answer="A2"),
                    make_acml_document(observation="Q3", answer="A3"),
                    make_acml_document(observation="Q4", answer="A4"),
                ],
                "bloom_level": ["remember", "understand", "remember", "understand"],
            }
        )

        sampled = training_dataset.resample_dataset_by_bloom_level(
            dataset,
            weights={"remember": 1.0, "understand": 0.0},
            seed=42,
        )

        self.assertEqual(len(sampled), len(dataset))
        self.assertEqual(set(sampled["bloom_level"]), {"remember"})

    def test_resample_dataset_by_bloom_level_requires_column_when_enabled(self) -> None:
        dataset = make_acml_dataset(make_acml_document())

        with self.assertRaisesRegex(ValueError, "没有 bloom_level 列"):
            training_dataset.resample_dataset_by_bloom_level(
                dataset,
                weights={"remember": 4.0},
                seed=42,
            )

    def test_build_train_dataset_skips_bloom_validation_when_sampling_disabled(self) -> None:
        raw_dataset = Dataset.from_dict(
            {
                "acml": [make_acml_document()],
                "bloom_level": [""],
            }
        )
        encoded_dataset = Dataset.from_dict({"input_ids": [[1, 2]], "labels": [[-100, 2]]})

        with patch("train_sft.load_dataset_source", return_value=raw_dataset), patch(
            "train_sft.prepare_training_dataset",
            return_value=encoded_dataset,
        ):
            dataset = train_sft.build_train_dataset(
                train_sft.ScriptArguments(dataset_path="/tmp/unit-test"),
                AgenticContextEncoder(FakeTokenizer()),
            )

        self.assertEqual(dataset.column_names, ["input_ids", "labels"])

    def test_encode_training_dataset_removes_source_columns(self) -> None:
        dataset = make_acml_dataset(
            make_acml_document(observation="Explain SFT", answer="Answer one"),
            make_acml_document(observation="Explain LoRA", answer="Answer two"),
        )

        encoded = encode_training_dataset(
            dataset,
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(),
        )

        self.assertEqual(len(encoded), 2)
        self.assertEqual(encoded.column_names, ["input_ids", "labels"])

    def test_prepare_training_dataset_allows_explicit_zero_limit(self) -> None:
        dataset = make_acml_dataset(make_acml_document())

        encoded = prepare_training_dataset(
            dataset,
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(),
            build_options=make_build_options(limit_train_samples=0),
        )

        self.assertEqual(len(encoded), 0)

    def test_prepare_training_dataset_accepts_jsonl_shard_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for shard_name, document in {
                "analyze--0000": make_acml_document(observation="Explain SFT", answer="Answer one"),
                "apply--0000": make_acml_document(observation="Explain LoRA", answer="Answer two"),
            }.items():
                shard_dir = root / shard_name
                shard_dir.mkdir()
                (shard_dir / "offsets.i32").write_bytes(b"")
                (shard_dir / "data.jsonl").write_text(
                    json.dumps({"sample_id": shard_name, "acml": document}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            dataset = load_dataset_source(dataset_path=str(root))
            encoded = prepare_training_dataset(
                dataset,
                encoder=AgenticContextEncoder(FakeTokenizer()),
                encoding_config=make_encoding_config(),
            )

        self.assertEqual(len(encoded), 2)
        self.assertEqual(encoded.column_names, ["input_ids", "labels"])

    def test_prepare_training_dataset_rejects_negative_limit(self) -> None:
        dataset = make_acml_dataset(make_acml_document())

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            prepare_training_dataset(
                dataset,
                encoder=AgenticContextEncoder(FakeTokenizer()),
                encoding_config=make_encoding_config(),
                build_options=make_build_options(limit_train_samples=-1),
            )

    def test_prepare_training_dataset_applies_limit_to_acml_rows(self) -> None:
        dataset = make_acml_dataset(
            make_acml_document(observation="Question 1", answer="Answer 1"),
            make_acml_document(observation="Question 2", answer="Answer 2"),
        )

        encoded = prepare_training_dataset(
            dataset,
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(),
            build_options=make_build_options(
                limit_train_samples=1,
                cache_dir=Path(tempfile.mkdtemp()),
            ),
        )

        self.assertEqual(len(encoded), 1)

    def test_prepare_training_dataset_reuses_cached_encoded_dataset(self) -> None:
        dataset = make_acml_dataset(make_acml_document())
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
        dataset = make_acml_dataset(make_acml_document())
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
        dataset = make_acml_dataset(make_acml_document())

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
        dataset = make_acml_dataset(make_acml_document())
        encoder = AgenticContextEncoder(FakeTokenizer())

        entry_key = training_dataset_cache_key(
            build_training_dataset_cache_identity(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(label_policy="entry"),
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

        self.assertNotEqual(entry_key, payload_key)

    def test_build_training_dataset_cache_identity_records_acml_protocol(self) -> None:
        cache_identity = build_training_dataset_cache_identity(
            make_acml_dataset(make_acml_document()),
            encoder=AgenticContextEncoder(FakeTokenizer()),
            encoding_config=make_encoding_config(),
            dataset_locator=make_dataset_locator(dataset_name="unit-test"),
        )

        self.assertEqual(cache_identity["data_protocol"], "acml")
        self.assertNotIn("dataset_format", cache_identity)
        self.assertNotIn("belief_prompt", cache_identity)

    def test_prepare_training_dataset_ignores_incomplete_cache_dir(self) -> None:
        dataset = make_acml_dataset(make_acml_document())
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
        dataset = make_acml_dataset(make_acml_document())
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

    def test_prepare_training_dataset_rejects_dataset_without_acml_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects a dataset with an 'acml' column"):
            prepare_training_dataset(
                Dataset.from_dict({"text": ["not acml"]}),
                encoder=AgenticContextEncoder(FakeTokenizer()),
                encoding_config=make_encoding_config(),
            )

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

    def test_load_dataset_source_supports_single_acml_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.acml"
            path.write_text(
                '<acml version="0"><acml:entry kind="observation">x</acml:entry></acml>',
                encoding="utf-8",
            )
            dataset = load_dataset_source(dataset_path=str(path))

        self.assertEqual(dataset.column_names, ["acml"])
        self.assertEqual(
            dataset[0]["acml"],
            '<acml version="0"><acml:entry kind="observation">x</acml:entry></acml>',
        )


if __name__ == "__main__":
    unittest.main()
