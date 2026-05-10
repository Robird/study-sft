"""Validate ACML records before preview, packaging, or training."""

from __future__ import annotations

import argparse

from study_sft.acml_dataset_utils import DEFAULT_ALLOWED_KINDS, validate_acml_records
from study_sft.cli_args import add_dataset_source_args
from study_sft.loaders import load_dataset_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_source_args(parser)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow_unsupervised", action="store_true")
    parser.add_argument("--allowed_kinds", nargs="+", default=list(DEFAULT_ALLOWED_KINDS))
    parser.add_argument("--max_reported_issues", type=int, default=20)
    return parser.parse_args()


def validate_acml_dataset(
    *,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    limit: int | None = None,
    allow_unsupervised: bool = False,
    allowed_kinds: list[str] | None = None,
    max_reported_issues: int = 20,
):
    dataset = load_dataset_source(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
    )
    rows = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    return validate_acml_records(
        rows,
        allowed_kinds=tuple(allowed_kinds or DEFAULT_ALLOWED_KINDS),
        require_supervision=not allow_unsupervised,
        max_issues=max_reported_issues,
    )


def main() -> None:
    args = parse_args()
    report = validate_acml_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        limit=args.limit,
        allow_unsupervised=args.allow_unsupervised,
        allowed_kinds=args.allowed_kinds,
        max_reported_issues=args.max_reported_issues,
    )
    print(
        "\n".join(
            [
                "ACML dataset validation report",
                f"total_records: {report.total_records}",
                f"valid_records: {report.valid_records}",
                f"invalid_records: {report.invalid_records}",
                f"supervised_records: {report.supervised_records}",
                f"unsupervised_records: {report.unsupervised_records}",
            ]
        )
    )
    if report.invalid_records:
        print("\nIssues:")
        for issue in report.issues:
            print(f"- {issue.source}: {issue.message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
