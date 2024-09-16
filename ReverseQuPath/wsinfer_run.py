import os
from pathlib import Path
from wsinfer import wsi
# from wsinfer.cli.infer import _print_system_info, _get_info_for_save
from wsinfer_zoo.client import HFModel
from wsinfer.modellib import models
import json
from wsinfer_zoo.client import ModelConfiguration
from wsinfer.patchlib import segment_and_patch_directory_of_slides
# from datetime import datetime
from openslide import OpenSlide
import argparse
from codecarbon import track_emissions

from run_inference import run_inference
from write_geojson import write_geojsons


def run(wsi_dir: Path = Path("WSIs"),
        results_dir: Path = Path("results"),
        config: Path | None = Path("config.json"),
        model_path: Path | None = Path("../weights/traced_model.pt"),
        batch_size: int = 16,  # set it according to the available GPU VRAM
        num_workers: int = 0,
        speedup: bool = False,
        pixel_classification: bool = False):

    wsi_dir = wsi_dir.resolve()
    results_dir = results_dir.resolve()

    if not wsi_dir.exists():
        raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

    files_in_wsi_dir = [p for p in wsi_dir.iterdir() if p.is_file()]
    if not files_in_wsi_dir:
        raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")

    wsi_suffix = dict([(p.stem, p.suffix) for p in files_in_wsi_dir])
    wsi_bounds = {}
    for p in files_in_wsi_dir:
        with OpenSlide(p) as slide:
            # print(slide.properties)
            try:
                wsi_bounds |= {p.stem: (
                    int(slide.properties["openslide.bounds-x"]),
                    int(slide.properties["openslide.bounds-y"]),
                    int(slide.properties["openslide.bounds-height"]),
                    int(slide.properties["openslide.bounds-width"]),
                )}
            except Exception as e:
                wsi_bounds |= {p.stem: (0, 0, 0, 0)}
                print(f"Slide properties not found: {e}")

    # _print_system_info()

    model_obj: HFModel | models.LocalModelTorchScript

    metadata = None
    if config is not None:
        with open(config) as f:
            _config_dict = json.load(f)
            class_decimals = tuple([-int(x.split("#")[1], 16) for x in _config_dict["class_hexadecimals"]])
        model_config = ModelConfiguration.from_dict(_config_dict)
        metadata = {
            "patch_size_pixels": model_config.patch_size_pixels,
            "class_names": model_config.class_names,
            "class_decimals": class_decimals,
            "wsi_suffix": wsi_suffix,
            "wsi_bounds": wsi_bounds,
        }
        # print("\nClass metadata:", metadata)
        model_obj = models.LocalModelTorchScript(
            config=model_config, model_path=str(model_path)
        )
        del _config_dict, model_config

    print("\nFinding patch coordinates...\n")
    segment_and_patch_directory_of_slides(
        wsi_dir=wsi_dir,
        save_dir=results_dir,
        patch_size_px=model_obj.config.patch_size_pixels,
        patch_spacing_um_px=model_obj.config.spacing_um_px,
        thumbsize=(2048, 2048),
        median_filter_size=7,
        binary_threshold=7,
        closing_kernel_size=6,
        min_object_size_um2=200 ** 2,
        min_hole_size_um2=190 ** 2,
    )

    print("\nRunning model inference...\n")
    failed_patching, failed_inference = run_inference(
        wsi_dir=wsi_dir,
        results_dir=results_dir,
        model_info=model_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        speedup=speedup,
        pixel_classification=pixel_classification,
        discard_last_class=True,
    )
    if failed_patching:
        print(f"\nPatching failed for {len(failed_patching)} slides")
        print("\n".join(failed_patching))
    if failed_inference:
        print(f"\nInference failed for {len(failed_inference)} slides")
        print("\n".join(failed_inference))

    # timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    # run_metadata_outpath = results_dir / f"run_metadata_{timestamp}.json"
    # print(f"Saving metadata about run to {run_metadata_outpath}")
    # run_metadata = _get_info_for_save(model_obj)
    # with open(run_metadata_outpath, "w") as f:
    #     json.dump(run_metadata, f, indent=2)

    print("Finished.")

    if not pixel_classification:
        csvs = list((results_dir / "model-outputs-csv").glob("*.csv"))
        write_geojsons(csvs, results_dir)
    else:
        npzs = list((results_dir / "model-outputs-npz").glob("*.npz"))
        npzs = [x for x in npzs if x.stem in wsi_suffix.keys()]
        for npz in npzs:
            write_geojsons([npz], results_dir, pixel_classification, metadata)


@track_emissions(offline=True, country_iso_code="ITA")
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    wsi.set_backend("openslide")
    run(wsi_dir=Path(args.wsi_dir), results_dir=Path(args.results_dir),
        config=Path(args.config), model_path=Path(args.model_path),
        batch_size=args.batch_size, speedup=args.speedup,
        pixel_classification=args.pixel_classification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run segmentation and classification of thyroid resection (HE-stained) cell nuclei at WSI level"
    )
    parser.add_argument(
        "--wsi-dir",
        default="WSIs",
        type=str,
        help="WSIs for model inference (default: WSIs)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=str,
        help="folder to save results (default: results)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        type=str,
        help="specify information required to run a patch/pixel classification model (default: config.json)",
    )
    parser.add_argument(
        "--model-path",
        default="../weights/traced_model.pt",
        type=str,
        help="specify model weights (default: ../weights/traced_model.pt)"
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="jit-compile the model, this has a startup cost but model inference should be faster (default: False)",
    )
    parser.add_argument(
        "--pixel-classification",
        action="store_true",
        help="specify whether the model is pixel based or path/tile based and returns pixel classes (default: False)"
    )
    arguments = parser.parse_args()
    main(arguments)
