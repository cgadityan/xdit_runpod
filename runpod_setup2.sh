#!/bin/bash
set -e
conda init --all
conda deactivate
conda activate flux

# --- Install additional dependencies ---
pip install "xfuser[diffusers,flash-attn]"

# --- Check GPU availability ---
python gpu_check.py

# --- Login to Hugging Face ---
huggingface-cli login --token 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

# --- Run Inference using torchrun ---
torchrun --nproc_per_node=2 ./xdit_usp.py --model black-forest-labs/FLUX.1-Fill-dev --ulysses_degree 2 --ring_degree 1 --num_inference_steps 50 --size 1224,1664
