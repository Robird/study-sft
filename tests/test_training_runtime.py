from __future__ import annotations

import errno
import json
from pathlib import Path
import tempfile
import train_sft
import unittest
from contextlib import contextmanager
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


class _FakeTrainTokenizer(FakeTokenizer):
    eos_token = "<|endoftext|>"
    eos_token_id = 151643
    pad_token = "<|PAD_TOKEN|>"
    pad_token_id = 151662


class _FakeTrainModel:
    def __init__(self) -> None:
        self.printed_trainable_parameters = False

    def print_trainable_parameters(self) -> None:
        self.printed_trainable_parameters = True


def make_encoding_config(
    *,
    max_length: int = 128,
) -> TrainingEncodingConfig:
    return TrainingEncodingConfig(max_length=max_length)


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
        f'<acml:entry kind="observation">{observation}</acml:entry>'
        f"{belief_entry}"
        f'<acml:entry kind="me" loss="true">{answer}</acml:entry>'
        "</acml>"
    )


def make_acml_dataset(*documents: str) -> Dataset:
    return Dataset.from_dict({"acml": list(documents)})


class TrainingRuntimeTests(unittest.TestCase):
    def test_resolve_trainable_token_indices_defaults_to_all_structure_tokens(self) -> None:
        self.assertEqual(
            train_sft.resolve_trainable_token_indices(train_sft.ScriptArguments()),
            [
                QWEN3_AGENTIC_TOKEN_TABLE.entry_start,
                QWEN3_AGENTIC_TOKEN_TABLE.entry_end,
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_start,
                QWEN3_AGENTIC_TOKEN_TABLE.opaque_payload_end,
                QWEN3_AGENTIC_TOKEN_TABLE.action_start,
                QWEN3_AGENTIC_TOKEN_TABLE.action_end,
            ],
        )

    def test_resolve_trainable_token_indices_can_be_disabled(self) -> None:
        self.assertIsNone(
            train_sft.resolve_trainable_token_indices(
                train_sft.ScriptArguments(lora_train_structural_tokens=False)
            )
        )

    def test_load_model_and_tokenizer_passes_trainable_structure_token_indices(self) -> None:
        captured_kwargs = {}
        fake_model = _FakeTrainModel()
        fake_tokenizer = _FakeTrainTokenizer()

        class _FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                return fake_model, fake_tokenizer

            @staticmethod
            def get_peft_model(model, **kwargs):
                captured_kwargs.update(kwargs)
                return model

        with patch.object(train_sft, "_require_unsloth", return_value=(_FakeFastLanguageModel, lambda: True)):
            model, tokenizer = train_sft.load_model_and_tokenizer(
                train_sft.ScriptArguments(model_name_or_path="fake-model")
            )

        self.assertIs(model, fake_model)
        self.assertIs(tokenizer, fake_tokenizer)
        self.assertTrue(fake_model.printed_trainable_parameters)
        self.assertEqual(
            captured_kwargs["trainable_token_indices"],
            train_sft.DEFAULT_TRAINABLE_STRUCTURE_TOKEN_IDS,
        )

    def test_assert_matching_training_tokenizer_ignores_pad_token_difference(self) -> None:
        encoded_encoder = AgenticContextEncoder(FakeTokenizer())
        training_tokenizer = FakeTokenizer()
        training_tokenizer.pad_token_id = 999999

        train_sft.assert_matching_training_tokenizer(encoded_encoder, training_tokenizer)

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
                    json.dumps({"sample_id": shard_name, "text": document}, ensure_ascii=False) + "\n",
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
                "input_ids": [[QWEN3_AGENTIC_TOKEN_TABLE.entry_start]],
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
        self.assertIn("action_start", cache_identity["policy"]["token_table"])
        self.assertIn("action_end", cache_identity["policy"]["token_table"])

    def test_tokenizer_identity_payload_records_action_tokens(self) -> None:
        identity = tokenizer_identity_payload(AgenticContextEncoder(FakeTokenizer()))

        self.assertIn(QWEN3_AGENTIC_TOKEN_TABLE.action_start_text, identity["probe_encodings"])
        self.assertIn(QWEN3_AGENTIC_TOKEN_TABLE.action_end_text, identity["probe_encodings"])

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

    def test_prepare_training_dataset_rechecks_cache_after_waiting_for_lock(self) -> None:
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
            cache_key = training_dataset_cache_key(cache_payload)
            cache_path = cache_dir / cache_key
            encoded = encode_training_dataset(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(),
            )

            @contextmanager
            def populate_cache_during_lock(_cache_path: Path):
                self.assertEqual(_cache_path, cache_path)
                encoded.save_to_disk(str(cache_path))
                training_cache.write_training_dataset_cache_metadata(
                    cache_path,
                    {
                        **cache_payload,
                        "cache_key": cache_key,
                        "validated": False,
                    },
                )
                yield

            with patch(
                "study_sft.training_cache.training_dataset_cache_lock",
                side_effect=populate_cache_during_lock,
            ), patch(
                "study_sft.training_dataset.encode_training_dataset",
                side_effect=AssertionError("cache should be re-checked after waiting for lock"),
            ):
                cached = prepare_training_dataset(
                    dataset,
                    encoder=encoder,
                    encoding_config=make_encoding_config(),
                    build_options=make_build_options(cache_dir=cache_dir),
                    dataset_locator=make_dataset_locator(dataset_name="unit-test"),
                )

        self.assertEqual(len(cached), 1)

    def test_prepare_training_dataset_recovers_from_directory_not_empty_rename_race(self) -> None:
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
            cache_key = training_dataset_cache_key(cache_payload)
            cache_path = cache_dir / cache_key
            winner_dataset = encode_training_dataset(
                dataset,
                encoder=encoder,
                encoding_config=make_encoding_config(),
            )
            original_rename = Path.rename

            def rename_with_race(self: Path, target: Path):
                if self.parent == cache_dir and Path(target) == cache_path and not cache_path.exists():
                    winner_dataset.save_to_disk(str(cache_path))
                    training_cache.write_training_dataset_cache_metadata(
                        cache_path,
                        {
                            **cache_payload,
                            "cache_key": cache_key,
                            "validated": False,
                        },
                    )
                    raise OSError(errno.ENOTEMPTY, "Directory not empty")
                return original_rename(self, target)

            with patch("pathlib.Path.rename", new=rename_with_race):
                cached = prepare_training_dataset(
                    dataset,
                    encoder=encoder,
                    encoding_config=make_encoding_config(),
                    build_options=make_build_options(cache_dir=cache_dir),
                    dataset_locator=make_dataset_locator(dataset_name="unit-test"),
                )

        self.assertEqual(len(cached), 1)

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

            self.assertEqual(
                [path.name for path in cache_dir.iterdir() if path.is_dir()],
                [],
            )

    def test_prepare_training_dataset_rejects_dataset_without_acml_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects a dataset with an 'acml' column"):
            prepare_training_dataset(
                Dataset.from_dict({"text": ["not acml"]}),
                encoder=AgenticContextEncoder(FakeTokenizer()),
                encoding_config=make_encoding_config(),
            )

    def test_agentic_data_collator_pads_features_consistently(self) -> None:
        collator = AgenticDataCollator(pad_token_id=QWEN3_AGENTIC_TOKEN_TABLE.entry_end)
        batch = collator(
            [
                {"input_ids": [1, 2], "labels": [-100, 2]},
                {"input_ids": [3], "labels": [3]},
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, QWEN3_AGENTIC_TOKEN_TABLE.entry_end]])
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
