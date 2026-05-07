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
                "  bash scripts/preview_tiny.sh",
                "  bash scripts/train_tiny_smoke.sh",
                "",
                "Main training entrypoints:",
                "  bash scripts/train_alpaca_chatml.sh",
                "  bash scripts/train_alpaca_late_system.sh",
                "  bash scripts/train_alpaca_bora.sh",
                "",
                "Read README.md for the learning map and experiment design.",
            ]
        )
    )


if __name__ == "__main__":
    main()
