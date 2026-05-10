#!/usr/bin/env bash
set -euo pipefail

python src/infer_lora.py \
  --adapter_path /mnt/fast/LLM/study-sft/smoke-acml-lora \
  --load_in_4bit false \
#   --prompt "用两句话解释 SFT 和预训练的区别。" \
  "$@"
