#!/bin/bash
set -e

# Upgrade pip/setuptools
pip install --upgrade pip setuptools wheel

# Install GPU-specific wheels first
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
# pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html 
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# pip install git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation

# Now install the package (including gpu + dev extras)
pip install -e ".[dev]"
