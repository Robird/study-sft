from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datasets import load_from_disk

import pack_acml_dataset
import validate_acml_dataset
from study_sft.acml_dataset_utils import collect_acml_paths, validate_acml_records


VALID_SAMPLE = (
    '<acml version="0"><acml:entry kind="observation">question</acml:entry>'
    '<acml:entry kind="me" loss="true">answer</acml:entry></acml>'
)
UNSUPERVISED_SAMPLE = (
    '<acml version="0"><acml:entry kind="observation">question</acml:entry>'
    '<acml:entry kind="me">answer</acml:entry></acml>'
)
INVALID_ROLE_SAMPLE = (
    '<acml version="0"><acml:entry kind="tool">tool output</acml:entry>'
    '<acml:entry kind="me" loss="true">answer</acml:entry></acml>'
)


class ACMLDatasetToolsTests(unittest.TestCase):
    def test_collect_acml_paths_recursively_returns_a_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            second = root / "b.acml"
            first = nested / "a.acml"
            second.write_text(VALID_SAMPLE, encoding="utf-8")
            first.write_text(VALID_SAMPLE, encoding="utf-8")

            paths = collect_acml_paths([root])

        self.assertEqual([path.as_posix() for path in paths], sorted(path.as_posix() for path in paths))
        self.assertEqual(sorted(path.name for path in paths), ["a.acml", "b.acml"])

    def test_pack_acml_dataset_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_path = root / "sample.acml"
            output_path = root / "train.jsonl"
            sample_path.write_text(VALID_SAMPLE, encoding="utf-8")

            count, output_format = pack_acml_dataset.pack_acml_dataset(
                [str(sample_path)],
                output_path=str(output_path),
                include_source_path=True,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(output_format, "jsonl")
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["acml"], VALID_SAMPLE)
        self.assertTrue(payload["source_path"].endswith("sample.acml"))

    def test_pack_acml_dataset_writes_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_path = root / "sample.acml"
            output_path = root / "packed_dataset"
            sample_path.write_text(VALID_SAMPLE, encoding="utf-8")

            count, output_format = pack_acml_dataset.pack_acml_dataset(
                [str(sample_path)],
                output_path=str(output_path),
                output_format="dataset",
            )

            dataset = load_from_disk(str(output_path))

        self.assertEqual(count, 1)
        self.assertEqual(output_format, "dataset")
        self.assertEqual(dataset.column_names, ["acml"])
        self.assertEqual(dataset[0]["acml"], VALID_SAMPLE)

    def test_pack_acml_dataset_rejects_invalid_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_path = root / "bad.acml"
            sample_path.write_text(UNSUPERVISED_SAMPLE, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no supervised entries"):
                pack_acml_dataset.pack_acml_dataset(
                    [str(sample_path)],
                    output_path=str(root / "train.jsonl"),
                )

    def test_validate_acml_records_reports_invalid_kinds_and_unsupervised_rows(self) -> None:
        report = validate_acml_records(
            (
                {"acml": VALID_SAMPLE, "source_path": "ok.acml"},
                {"acml": UNSUPERVISED_SAMPLE, "source_path": "unsupervised.acml"},
                {"acml": INVALID_ROLE_SAMPLE, "source_path": "bad-role.acml"},
            ),
            max_issues=10,
        )

        self.assertEqual(report.total_records, 3)
        self.assertEqual(report.valid_records, 1)
        self.assertEqual(report.invalid_records, 2)
        self.assertEqual(report.supervised_records, 1)
        self.assertEqual(report.unsupervised_records, 1)
        self.assertEqual([issue.source for issue in report.issues], ["unsupervised.acml", "bad-role.acml"])

    def test_validate_acml_dataset_loads_jsonl_and_reports_valid_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "train.jsonl"
            dataset_path.write_text(json.dumps({"acml": VALID_SAMPLE}, ensure_ascii=False) + "\n", encoding="utf-8")

            report = validate_acml_dataset.validate_acml_dataset(
                dataset_path=str(dataset_path),
            )

        self.assertEqual(report.total_records, 1)
        self.assertEqual(report.valid_records, 1)
        self.assertEqual(report.invalid_records, 0)


if __name__ == "__main__":
    unittest.main()
