import os
import glob
import numpy as np
import torch
from transformers import AutoTokenizer

# Based on src/study_sft/loaders.py default
DEFAULT_MODEL_PATH = "/mnt/fast/LLM/Qwen3-1.7B-Base"

def get_records(path, limit=None):
    files = sorted(glob.glob(os.path.join(path, "*.acml")))
    if limit:
        files = files[:limit]
    records = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fb:
            records.append(fb.read())
    return records

def main():
    sample_path = "/repos/qa-dump/output/zh/runs/help_gate_acml--prod-hg-1/artifacts/samples"
    records = get_records(sample_path, 1000)

    if not records:
        print("No records found.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    lengths = [len(tokenizer.encode(r)) for r in records]

    stats = {
        "min": np.min(lengths),
        "p50": np.percentile(lengths, 50),
        "p90": np.percentile(lengths, 90),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99),
        "max": np.max(lengths),
        "avg": np.mean(lengths)
    }

    count_1024 = sum(1 for l in lengths if l > 1024)
    count_1536 = sum(1 for l in lengths if l > 1536)
    count_2048 = sum(1 for l in lengths if l > 2048)

    print("\nToken Length Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")

    print(f"\nSamples > 1024: {count_1024}")
    print(f"Samples > 1536: {count_1536}")
    print(f"Samples > 2048: {count_2048}")

if __name__ == "__main__":
    main()
