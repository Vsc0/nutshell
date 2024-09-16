#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate nutshell || { echo "Failed to activate Conda environment"; exit 1; }

python wsinfer_run.py \
  --wsi-dir WSIs \
  --results-dir results \
  --config config.json \
  --model-path ../weights/traced_model.pt \
  --batch-size 16 \
  --pixel-classification

rm -rf QuPathProject

python create_qupath_project.py \
  --wsi-dir WSIs \
  --geojson-dir results/model-outputs-geojson \
  --qupath-class-names "HP,NIFTP,PTC" \
  --qupath-class-colors "#029e73,#0173b2,#d55e00" \
  --qupath-project-name QuPathProject \
  --add-images-groovy-script QuPathScripts/thyroid_import_images.groovy \
  --add-geojsons-groovy-script QuPathScripts/thyroid_import_geojsons.groovy \
  --qupath-bin-dir /opt/QuPath/bin \
  --import-images \
  --import-geojsons
# please use colors that are suitable for color-blind individuals

conda deactivate
