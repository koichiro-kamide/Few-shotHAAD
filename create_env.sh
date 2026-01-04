#!/usr/bin/env bash
set -e

conda create -n haad python=3.8
conda activate haad
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt