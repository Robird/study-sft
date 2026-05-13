#!/usr/bin/env bash
set -euo pipefail

sample_path=$(mktemp --suffix=.acml)
trap 'rm -f "$sample_path"' EXIT

cat >"$sample_path" <<'EOF'
<acml version="0"><acml:sentence role="belief">You are a helpful, honest, and concise assistant.</acml:sentence><acml:sentence role="observation">Explain what supervised fine-tuning is in one sentence.</acml:sentence><acml:sentence role="me" loss="true">Supervised fine-tuning teaches a model from labeled input-output examples.</acml:sentence></acml>
EOF

python src/preview_data.py \
  --dataset_path "$sample_path" \
  --limit 1
