#!/usr/bin/env bash

pip install gdown

FILE_ID="1130gHSvNyJmii7f6pv5aY5IyQIWc3t7R"
gdown --id "$FILE_ID" -O dataset.tar.gz

mkdir -p data
tar -xzf dataset.tar.gz -C data
rm dataset.tar.gz