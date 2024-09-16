#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate nutshell || { echo "Failed to activate Conda environment"; exit 1; }

python splitter.py \
  --pixel-size 0.25 \
  --tile-size 512 \
  --overlap 0 \
  --qupath-class-names HP,NIFTP,PTC,Other \
  --dataset-dir ~/datasets/thyroid \
  --dataset-metadata ~/datasets/thyroid/thyroid_dataset_metadata.csv

conda deactivate
