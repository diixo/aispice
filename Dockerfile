# create image:
# docker build -t torch-gpu .

# run image:
# docker run --rm -it --gpus all --name torchcheck torch-gpu /bin/bash

# python3 --version

######################################################################

# Base on PyTorch with CUDA 12.1 to simplify GPU stack
#FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel AS base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

# Copy sources
COPY . /app
WORKDIR /app

# Python 3.10.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel


# Python deps (GPU-enabled torch is already present in base image)
RUN pip install -r requirements.txt && pip cache purge


# Install PyTorch (reinstall scecified version):
RUN pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Checking script:
RUN echo 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))' > /app/test_torch.py

# By default, run checking:
CMD ["python3.10", "test_torch.py"]


# Node 20 LTS for building the Svelte UI
# RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
#  && apt-get update && apt-get install -y --no-install-recommends nodejs \
#  && rm -rf /var/lib/apt/lists/*

# # Build Svelte app to static assets (legacy-peer-deps for svelte-json-tree)
# RUN cd svelte \
#  && npm ci --legacy-peer-deps \
#  && npx vite build

