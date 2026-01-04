#!/usr/bin/env bash

pip install gdown

FILE_ID="1n2iBoivKGJpyDGSs4RDMrLDNU88cGLSA"
gdown --id "$FILE_ID" -O checkpoints_archive.zip

mkdir -p checkpoints_paper
unzip checkpoints_archive.zip -d checkpoints_paper
rm checkpoints_archive.zip