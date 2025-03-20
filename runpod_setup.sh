#!/bin/bash
set -e

# Change to the workspace directory (assumed to be mounted from the persistent volume)
# cd /workspace'
# Assume xdit_runpod directory exists

pwd
cd ../workspace/xdit_runpod
pwd

python -m venv flux
source flux/bin/activate
pip install requirements.txt
# --- Install Miniconda on the persistent volume if not already installed ---
# if [ ! -d "/workspace/miniconda3" ]; then
#     wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
# fi

# Add the persistent Miniconda to PATH
# export PATH="/workspace/miniconda3/bin:$PATH"

# --- Setup Conda Environment ---
# conda env create -f requirements.yml -n flux

pip install "xfuser[diffusers,flash-attn]"

# --- Check GPU availability ---
python gpu_check.py

# --- Login to Hugging Face ---
huggingface-cli login --token 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

# --- Run Inference using torchrun ---
torchrun --nproc_per_node=2 ./xdit_usp.py --model black-forest-labs/FLUX.1-Fill-dev --ulysses_degree 2 --ring_degree 1 --num_inference_steps 50 --size 1224,1664

exit 
