#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

ACML_DATASET_PATH=${ACML_DATASET_PATH:-}
if [[ -z "${ACML_DATASET_PATH}" ]]; then
  echo "Set ACML_DATASET_PATH to a .acml file, a JSON/JSONL dataset with an 'acml' or ACML 'text' column, a bloom-level shard root directory, or a datasets directory containing that column." >&2
  exit 1
fi

cd "${repo_root}"

torchrun --nproc_per_node=2 src/train_sft.py \
  --dataset_path "${ACML_DATASET_PATH}" \
  --output_dir /mnt/fast/LLM/study-sft/qwen3-1.7b-acml-lora \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --optim adamw_torch \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --max_length 2048 \
  --load_in_4bit false \
  --report_to none \
  "$@"
