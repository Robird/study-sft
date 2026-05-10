"""Study SFT entrypoint.

This repository is a small post-training lab around Qwen3-1.7B-Base.
Run this file for the command map, then use the scripts under ``scripts/`` for
actual preview, training, and inference.
"""

from __future__ import annotations


def main() -> None:
    print(
        "\n".join(
            [
                "study-sft: LLM post-training starter lab",
                "",
                "Recommended first steps:",
                "  bash scripts/preview_acml_tiny.sh",
                "  bash scripts/train_acml_tiny_smoke.sh",
                "",
                "Main training entrypoints:",
                "  ACML_DATASET_PATH=/path/to/train.acml bash scripts/train_acml_dataset.sh",
                "",
                "Main inference entrypoint:",
                "  bash scripts/infer_smoke_adapter.sh",
                "",
                "ACML dataset tools:",
                "  python src/validate_acml_dataset.py --dataset_path /path/to/train.jsonl",
                "  python src/pack_acml_dataset.py samples/ --output_path train.jsonl",
                "",
                "Read README.md for the learning map and experiment design.",
            ]
        )
    )


if __name__ == "__main__":
    main()
