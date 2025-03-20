#!/bin/bash
set -e

export PATH="/workspace/miniconda3/bin:$PATH" 
source /workspace/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda init --all
# """ps aux | grep "sshd: root" | grep -v grep | awk '{print $2}' | xargs kill -9""",

conda activate flux
python /workspace/xdit_runpod/gpu_check.py 
huggingface-cli login --token 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 
torchrun --nproc_per_node=2 /workspace/xdit_runpod/xdit_usp.py --model black-forest-labs/FLUX.1-Fill-dev --ulysses_degree 2 --ring_degree 1 --num_inference_steps 50 --size 1224,1664