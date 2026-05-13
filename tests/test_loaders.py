from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
from study_sft.loaders import get_effective_pad_token_id, load_dataset_source


VALID_ACML = (
    '<acml version="0"><acml:entry kind="observation">question</acml:entry>'
    '<acml:entry kind="me">answer</acml:entry></acml>'
)


class _DummyTokenizer:
    def __init__(self, *, pad_token_id=None, eos_token_id=None) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class LoaderTests(unittest.TestCase):
    def test_get_effective_pad_token_id_keeps_zero_pad_token_id(self) -> None:
        tokenizer = _DummyTokenizer(pad_token_id=0, eos_token_id=42)
        self.assertEqual(get_effective_pad_token_id(tokenizer), 0)

    def test_get_effective_pad_token_id_falls_back_to_eos(self) -> None:
        tokenizer = _DummyTokenizer(pad_token_id=None, eos_token_id=42)
        self.assertEqual(get_effective_pad_token_id(tokenizer), 42)

    def test_load_dataset_source_reads_acml_column_from_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(
                json.dumps({"sample_id": "sample-1", "acml": VALID_ACML}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            dataset = load_dataset_source(dataset_path=str(path))

        self.assertEqual(dataset.column_names, ["sample_id", "acml"])
        self.assertEqual(dataset[0]["sample_id"], "sample-1")
        self.assertEqual(dataset[0]["acml"], VALID_ACML)

    def test_load_dataset_source_reads_shard_directory_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_shard = root / "analyze--0000"
            second_shard = root / "apply--0000"
            first_shard.mkdir()
            second_shard.mkdir()
            (first_shard / "offsets.i32").write_bytes(b"")
            (second_shard / "offsets.i32").write_bytes(b"")
            (first_shard / "data.jsonl").write_text(
                json.dumps({"sample_id": "sample-1", "acml": VALID_ACML}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (second_shard / "data.jsonl").write_text(
                json.dumps({"sample_id": "sample-2", "acml": VALID_ACML}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            dataset = load_dataset_source(dataset_path=str(root))

        self.assertEqual(dataset.column_names, ["sample_id", "acml"])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["acml"], VALID_ACML)
        self.assertEqual(dataset[1]["acml"], VALID_ACML)

    def test_load_dataset_source_reads_acml_column_from_hub_dataset(self) -> None:
        with patch(
            "study_sft.loaders.load_dataset",
            return_value=Dataset.from_dict({"sample_id": ["sample-1"], "acml": [VALID_ACML]}),
        ):
            dataset = load_dataset_source(dataset_name="unit-test")

        self.assertEqual(dataset.column_names, ["sample_id", "acml"])
        self.assertEqual(dataset[0]["sample_id"], "sample-1")
        self.assertEqual(dataset[0]["acml"], VALID_ACML)

    def test_load_dataset_source_rejects_jsonl_without_acml_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(
                json.dumps({"sample_id": "sample-1", "text": VALID_ACML}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "named 'acml'"):
                load_dataset_source(dataset_path=str(path))


if __name__ == "__main__":
    unittest.main()
