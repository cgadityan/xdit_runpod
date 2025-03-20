#!/bin/bash
set -e

# Change to the workspace directory (assumed to be mounted from the persistent volume)
# cd /workspace'
# Assume xdit_runpod directory exists

pwd
cd ../workspace/xdit_runpod
pwd

# --- Install Miniconda on the persistent volume if not already installed ---
if [ ! -d "/workspace/miniconda3" ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
fi

# Add the persistent Miniconda to PATH
export PATH="/workspace/miniconda3/bin:$PATH"

# --- Setup Conda Environment ---
conda env create -f requirements.yml -n flux

exit 
