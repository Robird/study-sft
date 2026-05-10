#!/usr/bin/env bash
set -euo pipefail

sample_path=$(mktemp --suffix=.acml)
trap 'rm -f "$sample_path"' EXIT

cat >"$sample_path" <<'EOF'
<acml version="0"><acml:sentence role="belief">You are a helpful, honest, and concise assistant.</acml:sentence><acml:sentence role="observation">Explain what supervised fine-tuning is in one sentence.</acml:sentence><acml:sentence role="me" loss="true">Supervised fine-tuning teaches a model from labeled input-output examples.</acml:sentence></acml>
EOF

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/train_sft.py \
  --dataset_path "$sample_path" \
  --output_dir /mnt/fast/LLM/study-sft/smoke-acml-lora \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --optim adamw_torch \
  --logging_steps 1 \
  --save_steps 5 \
  --save_total_limit 1 \
  --max_length 1024 \
  --validate_encoding true \
  --load_in_4bit false \
  --report_to none \
  "$@"
