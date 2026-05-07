#!/usr/bin/env bash
set -euo pipefail

python src/preview_data.py \
  --dataset_path examples/tiny_alpaca.jsonl \
  --dataset_format alpaca \
  --prompt_mode chatml \
  --limit 2

python src/preview_data.py \
  --dataset_path examples/tiny_alpaca.jsonl \
  --dataset_format alpaca \
  --prompt_mode late_system \
  --limit 1

python src/preview_data.py \
  --dataset_path examples/tiny_alpaca.jsonl \
  --dataset_format alpaca \
  --prompt_mode bora \
  --limit 1
