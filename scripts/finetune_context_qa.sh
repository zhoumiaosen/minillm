#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export HF_HOME="$PWD/.cache/huggingface"
python_bin="${PYTHON_BIN:-python3}"
data_dir="$PWD/runs/context-qa"
test -s "$data_dir/train.jsonl"
test -s "$data_dir/manifest.json"
test -s out/small384l8_full_sft_384.pth
# A fresh experiment must not overwrite an earlier run. Resume explicitly if needed.
if [[ -e out/small384l8_context_qa_384.pth || -e checkpoints/small384l8_context_qa_384_resume.pth ]]; then
    echo 'Context-QA output already exists; inspect it before starting or resuming.' >&2
    exit 1
fi
"$python_bin" train.py sft \
    --hidden_size 384 --num_hidden_layers 8 --use_moe 0 \
    --from_weight small384l8_full_sft --save_weight small384l8_context_qa \
    --data_path "$data_dir/train.jsonl" --max_seq_len 512 \
    --epochs 1 --batch_size 8 --accumulation_steps 1 --learning_rate 1e-5 \
    --dtype bfloat16 --device cuda:0 --num_workers 0 \
    --log_interval 50 --save_interval 250 --from_resume 0 \
    2>&1 | tee "$data_dir/train.log"
