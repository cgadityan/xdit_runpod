#!/bin/bash
set -e

# Change to the workspace directory (assumed to be mounted from the persistent volume)
# cd /workspace'
# Assume xdit_runpod directory exists
pwd
cd ../workspace/xdit_runpod
pwd

# Add the persistent Miniconda to PATH
export PATH="/workspace/miniconda3/bin:$PATH"

# --- Check GPU availability ---
python gpu_check.py

# --- Login to Hugging Face ---
huggingface-cli login --token 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

# --- Run Inference using torchrun ---
torchrun --nproc_per_node=2 ./xdit_usp.py --model black-forest-labs/FLUX.1-Fill-dev --ulysses_degree 2 --ring_degree 1 --num_inference_steps 50 --size 1224,1664

exit 
