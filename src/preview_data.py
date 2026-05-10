"""Preview how ACML rows become agentic-context contexts."""

from __future__ import annotations

import argparse
import json
from itertools import islice

from study_sft.adapters.acml import agentic_context_from_acml_record
from study_sft.agentic_context import AgenticContextEncoder
from study_sft.cli_args import add_dataset_source_args, add_model_source_args
from study_sft.loaders import load_base_tokenizer, load_dataset_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_source_args(parser)
    add_model_source_args(parser)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=2400)
    parser.add_argument("--show_token_text", action="store_true")
    parser.add_argument("--show_spans", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    needs_encoder = args.show_token_text or args.show_spans
    needs_debug_encoding = args.show_spans
    tokenizer = None
    encoder = None
    if needs_encoder:
        tokenizer = load_base_tokenizer(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
        encoder = AgenticContextEncoder(tokenizer)

    dataset = load_dataset_source(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
    )
    rows = islice(dataset, args.limit)

    for index, record in enumerate(rows, start=1):
        context = agentic_context_from_acml_record(dict(record))
        print(f"\n{'=' * 24} record {index} {'=' * 24}")
        print("\n[context]")
        print(json.dumps(context.to_dict(), ensure_ascii=False, indent=2))
        if not needs_encoder:
            continue

        assert encoder is not None
        assert tokenizer is not None

        debug_encoded = None
        encoded = None
        if needs_debug_encoding:
            debug_encoded = encoder.encode_context_with_debug(context, validate=True)
            encoded = debug_encoded.encoded
        elif args.show_token_text:
            encoded = encoder.encode_context(context, validate=True)

        if args.show_token_text:
            assert encoded is not None
            decoded = tokenizer.decode(encoded.input_ids, skip_special_tokens=False)
            print("\n[token_text]")
            print(decoded[: args.max_chars])
            if len(decoded) > args.max_chars:
                print(f"\n... truncated: {len(decoded) - args.max_chars} chars")

        if needs_debug_encoding:
            assert debug_encoded is not None
            print("\n[entry_spans]")
            print(json.dumps(debug_encoded.to_dict()["entry_spans"], ensure_ascii=False, indent=2))
            print("\n[spans]")
            print(json.dumps(debug_encoded.to_dict()["spans"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
