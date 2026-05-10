"""Helpers for packaging and validating ACML datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from acml import ACMLParseError, parse_document
from study_sft.adapters.acml import ACMLLossPolicy, agentic_context_from_acml_document


DEFAULT_ALLOWED_KINDS: tuple[str, ...] = ("belief", "observation", "me")


@dataclass(frozen=True, slots=True)
class ACMLValidationIssue:
    index: int
    source: str
    message: str


@dataclass(frozen=True, slots=True)
class ACMLValidationReport:
    total_records: int
    valid_records: int
    invalid_records: int
    supervised_records: int
    unsupervised_records: int
    issues: tuple[ACMLValidationIssue, ...]


def collect_acml_paths(inputs: Sequence[str | Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_input in inputs:
        path = Path(raw_input)
        if not path.exists():
            raise ValueError(f"input path does not exist: {path}")
        if path.is_file():
            if path.suffix != ".acml":
                raise ValueError(f"expected a .acml file, got: {path}")
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
            continue
        found = sorted(child.resolve() for child in path.rglob("*.acml") if child.is_file())
        if not found:
            raise ValueError(f"directory contains no .acml files: {path}")
        for child in found:
            if child not in seen:
                seen.add(child)
                paths.append(child)
    if not paths:
        raise ValueError("no ACML input files were collected")
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def records_from_acml_paths(
    paths: Sequence[Path],
    *,
    include_source_path: bool = False,
) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for path in paths:
        record = {"acml": path.read_text(encoding="utf-8")}
        if include_source_path:
            record["source_path"] = str(path)
        records.append(record)
    return tuple(records)


def validate_acml_records(
    records: Iterable[Mapping[str, Any]],
    *,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
    require_supervision: bool = True,
    loss_policy: ACMLLossPolicy = "explicit",
    max_issues: int | None = None,
) -> ACMLValidationReport:
    allowed_kind_set = set(allowed_kinds)
    total_records = 0
    valid_records = 0
    invalid_records = 0
    supervised_records = 0
    unsupervised_records = 0
    issues: list[ACMLValidationIssue] = []

    for index, record in enumerate(records):
        total_records += 1
        source = source_label_for_record(record, index=index)
        try:
            source_text = record.get("acml")
            if not isinstance(source_text, str) or not source_text:
                raise ValueError("record must contain a non-empty string 'acml' field")
            document = parse_document(source_text)
            invalid_kinds = sorted({entry.kind for entry in document.entries if entry.kind not in allowed_kind_set})
            if invalid_kinds:
                raise ValueError(f"unsupported kinds: {', '.join(invalid_kinds)}")
            context = agentic_context_from_acml_document(document, loss_policy=loss_policy)
            supervised = any(entry.loss for entry in context.entries)
            if supervised:
                supervised_records += 1
            else:
                unsupervised_records += 1
            if require_supervision and not supervised:
                raise ValueError("record has no supervised entries under the current loss policy")
        except (ACMLParseError, ValueError) as exc:
            invalid_records += 1
            if max_issues is None or len(issues) < max_issues:
                issues.append(ACMLValidationIssue(index=index, source=source, message=str(exc)))
            continue
        valid_records += 1

    return ACMLValidationReport(
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        supervised_records=supervised_records,
        unsupervised_records=unsupervised_records,
        issues=tuple(issues),
    )


def source_label_for_record(record: Mapping[str, Any], *, index: int) -> str:
    source_path = record.get("source_path")
    if isinstance(source_path, str) and source_path:
        return source_path
    return f"record[{index}]"
