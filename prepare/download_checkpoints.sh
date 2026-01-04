#!/usr/bin/env bash

pip install gdown

FILE_ID="1n2iBoivKGJpyDGSs4RDMrLDNU88cGLSA"
gdown --id "$FILE_ID" -O checkpoints.zip

unzip checkpoints.zip
mkdir -p checkpoints_paper
mv encoder.p humanmac.pt checkpoints_paper/
rm checkpoints.zip