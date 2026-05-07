#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/train_sft.py \
  --dataset_path examples/tiny_alpaca.jsonl \
  --dataset_format alpaca \
  --prompt_mode chatml \
  --output_dir /mnt/fast/LLM/study-sft/smoke-chatml-lora \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --optim adamw_torch \
  --logging_steps 1 \
  --save_steps 5 \
  --save_total_limit 1 \
  --max_length 1024 \
  --load_in_4bit false \
  --report_to none \
  "$@"
