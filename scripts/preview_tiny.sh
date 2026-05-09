#!/usr/bin/env bash
set -euo pipefail

python src/preview_data.py \
  --dataset_path examples/tiny_alpaca.jsonl \
  --dataset_format alpaca \
  --limit 2
