"""Pack one-file-per-sample ACML authoring files into a trainable dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from study_sft.acml_dataset_utils import collect_acml_paths, records_from_acml_paths, validate_acml_records


OUTPUT_FORMAT_CHOICES = ("jsonl", "dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="One or more .acml files or directories containing .acml files")
    parser.add_argument("--output_path", required=True, help="Output .jsonl path or dataset directory path")
    parser.add_argument("--output_format", choices=OUTPUT_FORMAT_CHOICES)
    parser.add_argument("--include_source_path", action="store_true")
    parser.add_argument("--allow_unsupervised", action="store_true")
    parser.add_argument("--max_reported_issues", type=int, default=20)
    return parser.parse_args()


def infer_output_format(output_path: Path, explicit_format: str | None) -> str:
    if explicit_format is not None:
        return explicit_format
    if output_path.suffix == ".jsonl":
        return "jsonl"
    return "dataset"


def pack_acml_dataset(
    inputs: list[str],
    *,
    output_path: str,
    output_format: str | None = None,
    include_source_path: bool = False,
    allow_unsupervised: bool = False,
    max_reported_issues: int = 20,
) -> tuple[int, str]:
    collected_paths = collect_acml_paths(inputs)
    records = records_from_acml_paths(collected_paths, include_source_path=include_source_path)
    report = validate_acml_records(
        records,
        require_supervision=not allow_unsupervised,
        max_issues=max_reported_issues,
    )
    if report.invalid_records:
        lines = [
            f"Found {report.invalid_records} invalid ACML record(s) while packing.",
        ]
        for issue in report.issues:
            lines.append(f"- {issue.source}: {issue.message}")
        raise ValueError("\n".join(lines))

    output = Path(output_path)
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    resolved_format = infer_output_format(output, output_format)
    if resolved_format == "jsonl":
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
    elif resolved_format == "dataset":
        dataset = Dataset.from_list(list(records))
        dataset.save_to_disk(str(output))
    else:
        raise ValueError(f"unsupported output_format: {resolved_format!r}")

    return len(records), resolved_format


def main() -> None:
    args = parse_args()
    count, resolved_format = pack_acml_dataset(
        args.inputs,
        output_path=args.output_path,
        output_format=args.output_format,
        include_source_path=args.include_source_path,
        allow_unsupervised=args.allow_unsupervised,
        max_reported_issues=args.max_reported_issues,
    )
    print(f"Packed {count} ACML record(s) into {resolved_format}: {args.output_path}")


if __name__ == "__main__":
    main()
