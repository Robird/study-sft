#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

cd "${repo_root}"

CUDA_VISIBLE_DEVICES=1 python src/infer_lora.py \
  --adapter_path /mnt/fast/LLM/study-sft/qwen3-1.7b-acml-lora \
  --load_in_4bit false \
  --debug_tokens \
  "$@"
