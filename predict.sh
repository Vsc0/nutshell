#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate nutshell || { echo "Failed to activate Conda environment"; exit 1; }

python predictor.py \
  --batch-size 8 \
  --partition test

conda deactivate
