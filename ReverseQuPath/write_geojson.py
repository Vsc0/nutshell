"""Convert CSVs of model outputs to GeoJSON files.

GeoJSON files can be loaded into whole slide image viewers like QuPath.
"""

from __future__ import annotations

import json
import uuid
# from functools import partial
from pathlib import Path
import pandas as pd
# from tqdm.contrib.concurrent import process_map
import numpy as np
import numpy.typing as npt
import rasterio.features


def _box_to_polygon(
        *, minx: int, miny: int, width: int, height: int
) -> list[tuple[int, int]]:
    """Get coordinates of a box polygon."""
    maxx = minx + width
    maxy = miny + height
    return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]


def _row_to_geojson(row: pd.Series, prob_cols: list[str]) -> dict:
    """Convert information about one tile to a single GeoJSON feature."""
    minx, miny, width, height = row["minx"], row["miny"], row["width"], row["height"]
    coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height)
    prob_dict = row[prob_cols].to_dict()

    measurements = {}
    for k, v in prob_dict.items():
        measurements[k] = v

    return {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "isLocked": True,
            # measurements is a list of {"name": str, "value": float} dicts.
            # https://qupath.github.io/javadoc/docs/qupath/lib/measurements/MeasurementList.html
            "measurements": measurements,
            "objectType": "tile"
            # classification is a dict of "name": str and optionally "color": int.
            # https://qupath.github.io/javadoc/docs/qupath/lib/objects/classes/PathClass.html
            # We do not include classification because we do not enforce a single class
            # per tile.
            # "classification": {"name": class_name},
        },
    }


def _arr_to_geojson(coords_arr: npt.NDArray[np.int_], classes_arr: npt.NDArray[np.int_],
                    metadata: dict, wsi_suffix: str, wsi_bounds: tuple[int]) -> list:
    """Convert information about one tile to a single GeoJSON features."""
    bounds_x, bounds_y, bounds_height, bounds_width = wsi_bounds
    patch_size_pixels = metadata["patch_size_pixels"]
    minx, miny, width, height = int(coords_arr[0]), int(coords_arr[1]), int(coords_arr[2]), int(coords_arr[3])
    if wsi_suffix in (".svs", ".ndpi"):
        transform = rasterio.transform.from_origin(minx, miny, width / patch_size_pixels, - height / patch_size_pixels)
    elif wsi_suffix == ".mrxs":
        transform = rasterio.transform.from_origin(minx - bounds_x,
                                                   miny - bounds_y,
                                                   width / patch_size_pixels,
                                                   - height / patch_size_pixels)

    shapes = rasterio.features.shapes(source=classes_arr, mask=classes_arr > 0, connectivity=4, transform=transform)

    classes = metadata["class_names"]
    decimals = metadata["class_decimals"]

    features = []
    for polygon, value in shapes:
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            # "id": "PathDetectionObject",
            "geometry": polygon,
            "properties": {
                "isLocked": True,
                "measurements": [{'name': 'Label', 'value': value}],
                "objectType": "detection",
                # "objectType": "annotation",
                "classification": {"name": classes[int(value) - 1],
                                   "colorRGB": decimals[int(value) - 1]},
            }
        }
        features.append(feature)
    return features


def _dataframe_to_geojson(df: pd.DataFrame, prob_cols: list[str]) -> dict:
    """Convert a dataframe of tiles to GeoJSON format."""
    features = df.apply(_row_to_geojson, axis=1, prob_cols=prob_cols)
    return {
        "type": "FeatureCollection",
        "features": features.tolist(),
    }


def _npz_to_geojson(slide_coords_arr: npt.NDArray[np.uint32],
                    slide_classes_arr: npt.NDArray[np.uint8],
                    metadata: dict, wsi_suffix: str, wsi_bounds: tuple[int]) -> dict:
    """Convert a numpy array of tiles pixels to GeoJSON format."""
    features = []
    for tile_coords_arr, tile_classes_arr in zip(slide_coords_arr, slide_classes_arr):
        features.extend(_arr_to_geojson(tile_coords_arr, tile_classes_arr, metadata, wsi_suffix, wsi_bounds))
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def make_geojson(csv: Path, results_dir: Path,
                 pixel_classification: bool = False, metadata: dict = None) -> None:
    filename = csv.stem
    print(f"Converting {filename} to GeoJSON")
    if not pixel_classification:
        df = pd.read_csv(csv)
        prob_cols = [col for col in df.columns.tolist() if col.startswith("prob_")]
        if not prob_cols:
            raise KeyError("Did not find any columns with prob_ prefix.")
        geojson = _dataframe_to_geojson(df, prob_cols)
    else:
        with np.load(csv) as data:
            slide_coords_arr = data["slide_coords_arr"].astype(np.uint32)
            if pixel_classification:
                slide_classes_arr = data["slide_classes_arr"].astype(np.uint8)
        wsi_suffix: str = metadata['wsi_suffix'][filename]
        wsi_bounds: tuple[int] = metadata['wsi_bounds'][filename]
        if pixel_classification:
            geojson = _npz_to_geojson(slide_coords_arr, slide_classes_arr, metadata, wsi_suffix, wsi_bounds)

    with open(results_dir / "model-outputs-geojson" / f"{filename}.json", "w") as f:
        json.dump(geojson, f)


def write_geojsons(csvs: list[Path], results_dir: Path,
                   pixel_classification: bool = False, metadata: dict = None) -> None:
    output = results_dir / "model-outputs-geojson"

    ext = 'csv' if not pixel_classification else 'npz'

    if not results_dir.exists():
        raise FileExistsError(f"results_dir does not exist: {results_dir}")
    if (
            not (results_dir / f"model-outputs-{ext}").exists()
            and (results_dir / "patches").exists()
    ):
        raise FileExistsError(
            "Model outputs have not been generated yet. Please run model inference."
        )
    if not (results_dir / f"model-outputs-{ext}").exists():
        raise FileExistsError(
            f"Expected results_dir to contain a 'model-outputs-{ext}' "
            "directory but it does not."
            "Please provide the path to the directory"
            "that contains model-outputs, masks, and patches."
        )
    if output.exists():
        geojsons = list((results_dir / "model-outputs-geojson").glob("*.json"))
        # make a list of filenames for both geojsons and csvs
        geojson_filenames = [filename.stem for filename in geojsons]
        csv_filenames = [filename.stem for filename in csvs]
        # make a list of new csvs that need to be converted to geojson
        csvs_new = [csv for csv in csv_filenames if csv not in geojson_filenames]
        csvs = [path for path in csvs if path.stem in csvs_new]
    else:
        # if output directory doesn't exist, make one and set csvs_final to csvs
        output.mkdir(parents=True, exist_ok=True)

    # func = partial(make_geojson, results_dir=results_dir, pixel_classification=pixel_classification, metadata=metadata)
    # process_map(func, csvs, max_workers=1)
    for csv in csvs:
        make_geojson(csv, results_dir, pixel_classification, metadata)
