"""Preview how dataset rows become SFT training text."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from datasets import IterableDataset, load_dataset, load_from_disk

from study_sft.formats import (
    DEFAULT_BORA_REASONING,
    DEFAULT_SYSTEM_PROMPT,
    format_sft_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config")
    parser.add_argument("--dataset_path")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--dataset_format", choices=["alpaca", "messages", "sharegpt", "text"], default="alpaca")
    parser.add_argument("--prompt_mode", choices=["chatml", "late_system", "bora"], default="chatml")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--bora_reasoning", default=DEFAULT_BORA_REASONING)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=2400)
    return parser.parse_args()


def load_any_dataset(args: argparse.Namespace):
    if args.dataset_path:
        path = Path(args.dataset_path)
        if path.is_file():
            return load_dataset("json", data_files=str(path), split=args.dataset_split)
        return load_from_disk(str(path))
    if not args.dataset_name:
        raise SystemExit("需要 --dataset_path 或 --dataset_name")
    return load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )


def main() -> None:
    args = parse_args()
    dataset = load_any_dataset(args)
    rows = islice(dataset, args.limit) if isinstance(dataset, IterableDataset) else dataset.select(range(min(args.limit, len(dataset))))

    for index, record in enumerate(rows, start=1):
        text = format_sft_text(
            dict(record),
            dataset_format=args.dataset_format,
            prompt_mode=args.prompt_mode,
            default_system=args.system_prompt,
            bora_reasoning=args.bora_reasoning,
        )
        print(f"\n{'=' * 24} sample {index} {'=' * 24}")
        print(text[: args.max_chars])
        if len(text) > args.max_chars:
            print(f"\n... truncated: {len(text) - args.max_chars} chars")


if __name__ == "__main__":
    main()
