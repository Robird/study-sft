#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=2 src/train_sft.py \
  --dataset_name yahma/alpaca-cleaned \
  --dataset_format alpaca \
  --prompt_mode chatml \
  --output_dir /mnt/fast/LLM/study-sft/qwen3-1.7b-alpaca-chatml-lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --optim adamw_torch \
  --max_steps 500 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --load_in_4bit false \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --gradient_checkpointing \
  --bf16 \
  --report_to none \
  "$@"
