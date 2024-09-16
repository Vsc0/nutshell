from pathlib import Path
from os import environ
environ["PAQUO_QUPATH_DIR"] = '/opt/QuPath'
# paquo config --search-tree
# paquo config -l
try:
    from paquo.projects import QuPathProject
except ImportError:
    print("""If QuPath is installed, please define the environment variable 
             PAQUO_QUPATH_DIR with the location of the QuPath installation. 
             If QuPath is not installed, please install it from https://qupath.github.io/.""")
from paquo.classes import QuPathPathClass
import subprocess
import os
import argparse


def add_images(args):
    image_dir = Path(args.wsi_dir).resolve()
    # add images via groovy because it is way faster than PAQUO
    os.environ["PATH"] += ":" + args.qupath_bin_dir
    qupath_project_path = Path(os.path.join(args.qupath_project_name, "project.qpproj")).resolve()
    subprocess.run([
        "QuPath",
        # "--log=DEBUG",
        "script",
        args.add_images_groovy_script,
        "--save",
        "--args",
        f"[{image_dir},{qupath_project_path}]",
    ], check=True)


def add_annotations(args):
    image_dir = Path(args.wsi_dir).resolve()
    geojson_dir = Path(args.geojson_dir).resolve()
    # add detections via a groovy script because it is way faster than PAQUO
    os.environ["PATH"] += ":" + args.qupath_bin_dir
    qupath_project_path = Path(os.path.join(args.qupath_project_name, "project.qpproj")).resolve()
    subprocess.run([
        "QuPath",
        # "--log=DEBUG",
        "script",
        args.add_geojsons_groovy_script,
        "--save",
        "--args",
        f"[{image_dir},{geojson_dir},{qupath_project_path}]",
    ], check=True)


def main(args, verbose: bool = False) -> None:
    class_names = args.qupath_class_names
    class_colors = args.qupath_class_colors
    if not os.path.isdir(args.qupath_project_name):
        with QuPathProject(args.qupath_project_name, mode="a") as qp:
            new_classes = []
            for class_name, class_color in zip(class_names, class_colors):
                new_classes.append(QuPathPathClass(name=class_name, color=class_color))
            qp.path_classes = new_classes  # setting QuPathProject.path_class always replaces all classes
            if verbose:
                print('qp.images', qp.images)

    if args.import_images:
        add_images(args)
    if args.import_geojsons:
        add_annotations(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run segmentation and classification of thyroid resection cell nuclei at (HE-stained) WSI level"
    )
    parser.add_argument(
        "--wsi-dir",
        default="WSI",
        type=str,
        help="WSI for model inference (default: WSI)",
    )
    parser.add_argument(
        "--geojson-dir",
        default="results/model-outputs-geojson",
        type=str,
        help="folder to load geojson objects (default: results/model-outputs-geojson)",
    )

    def tuple_of_strings(arg):
        return tuple(arg.split(','))

    parser.add_argument(
        '--qupath-class-names',
        default='Hyperplastic nodule,'
                'Non-Invasive Follicular Tumor with Papillary-like nuclear features,'
                'Papillary Thyroid Carcinoma',
        type=tuple_of_strings,
        help='QuPath class names used at detection time',
    )
    parser.add_argument(
        '--qupath-class-colors',
        default='#029e73,#0173b2,#d55e00',
        type=tuple_of_strings,
        help='QuPath class colors used at detection time',
    )
    parser.add_argument(
        "--qupath-project-name",
        default="QuPathProject",
        type=str,
        help="QuPath project name (default: QuPathProject)",
    )
    parser.add_argument(
        '--add-images-groovy-script',
        default='QuPathScripts/thyroid_import_images.groovy',
        type=str,
        help='groovy script to add images to the QuPath project',
    )
    parser.add_argument(
        '--add-geojsons-groovy-script',
        default='QuPathScripts/thyroid_import_geojsons.groovy',
        type=str,
        help='groovy script to add geojsons to the QuPath project',
    )
    parser.add_argument(
        "--qupath-bin-dir",
        default='/opt/QuPath/bin',
        type=str,
        help='QuPath bin directory (default: /opt/QuPath/bin)',
    )
    parser.add_argument(
        "--import-images",
        action='store_true',
        help='whether to import the images (default: False)',
    )
    parser.add_argument(
        "--import-geojsons",
        action='store_true',
        help='whether to import the geojsons (default: False)',
    )
    arguments = parser.parse_args()
    main(arguments, verbose=True)
