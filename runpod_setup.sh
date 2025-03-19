mkdir cgadityan
cd cgadityan
mkdir data
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# new terminal session
# import files, data, images
#!/bin/bash
gnome-terminal -- bash -c "echo 'Hello from new terminal'; ls; exec bash"


git clone https://github.com/xdit-project/xDiT.git
conda env create -f requirements.yml
conda deactivate
conda activate flux
pip install "xfuser[diffusers,flash-attn]"
python gpu_check.py
huggingface-cli login --token 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
torchrun --nproc_per_node=2 ./xdit_usp.py --model black-forest-labs/FLUX.1-Fill-dev --ulysses_degree 2 --ring_degree 1 --num_inference_steps 50 --size 1224,1664

# rpa_AEEUIB2T6MOZML2PN4DQ2YWR4H8CZYIZRKFOJUO61q26jg

