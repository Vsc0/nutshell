#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate nutshell || { echo "Failed to activate Conda environment"; exit 1; }

cd MorphometricAnalysis || { echo "Failed to change directory"; exit 1; }

python morphometric_features.py \
  --qupath-exported-measurements data/measurements_1um_per_pixel.csv

python molecular_features.py

conda deactivate
