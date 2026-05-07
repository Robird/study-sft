#!/usr/bin/env bash
set -euo pipefail

python src/infer_lora.py \
  --adapter_path /mnt/fast/LLM/study-sft/smoke-chatml-lora \
  --load_in_4bit false \
  --prompt_mode chatml \
  --prompt "用两句话解释 SFT 和预训练的区别。" \
  "$@"
