#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate nutshell || { echo "Failed to activate Conda environment"; exit 1; }

python trainer.py \
  --batch-size 8 \
  --early-stopping

conda deactivate
