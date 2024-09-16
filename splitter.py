import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils import pd2csv


def tile_annotator(tile_dir: str, class_names: list[str], dataset_metadata: str) -> list[dict]:
    print('dataset_metadata', dataset_metadata)
    df = pd.read_csv(dataset_metadata)
    df['wsi'] = df['wsi'].apply(lambda x: x.split('.')[0])
    dataset_dicts = []
    path = tile_dir
    with os.scandir(path) as it:
        for i, entry in enumerate(it):
            if entry.is_dir():
                wsi_name = entry.name
                path = os.path.join(tile_dir, wsi_name)
                label_info = os.path.join(path, f'{wsi_name}-tiles.json')
                if os.path.isfile(label_info):
                    with open(label_info) as stream:
                        info = json.load(stream)
                        base_dir = info['base_directory']
                        for tile in tqdm(info['tiles'], desc=f'Processing tiles of WSI {wsi_name}'):
                            tile_name = os.path.join(base_dir, tile['image'])
                            mask_name = os.path.join(base_dir, tile['labels'])

                            record = {
                                'tile_name': tile_name,
                                'wsi': wsi_name,
                                'wsi_downsample': tile['region']['downsample'],
                                'wsi_x': tile['region']['x'],
                                'wsi_y': tile['region']['y'],
                                'wsi_h': tile['region']['height'],
                                'wsi_w': tile['region']['width'],
                                'image_id': tile['image'].split('/')[-1].split('.')[0],
                                'sem_seg_file_name': mask_name,
                            }

                            mask = Image.open(mask_name)
                            target = np.array(mask)
                            mask.close()
                            class_indices = np.unique(target)
                            class_indices.sort()
                            for class_index in class_indices:
                                class_index_mask = np.equal(class_index, target)
                                record.update({str(class_index): np.sum(class_index_mask)})

                            partially_annotated = False
                            if '255' in record.keys():
                                if record['255'] > 0:
                                    partially_annotated = True
                                    target = np.where(target == 255, 0, target)
                                    cv2.imwrite(f'{mask_name[:-4]}_sparse{mask_name[-4:]}', target)
                            record.update({'partially_annotated': partially_annotated})

                            majority_class = 0
                            maximum_px_count = 0
                            for class_index in class_indices:
                                if class_index in range(1, len(class_names) + 1):
                                    class_px_count = record[str(class_index)]
                                    if class_px_count > maximum_px_count:
                                        maximum_px_count = class_px_count
                                        majority_class = class_index
                            record.update({'majority_class': class_names[majority_class]})

                            row = df[df['wsi'].eq(wsi_name)].values[0]
                            record.update({'project': row[1], 'center': row[2].lower(), 'patient_id': row[3]})

                            dataset_dicts.append(record)
    return dataset_dicts


def compute_class_weights(df, len_classes, partition, class_weights):
    weight = df[map(str, range(len_classes + 1))].sum().values
    weight_min = weight.min()
    weight = 2 - (weight - weight_min) / (weight.max() - weight_min)
    class_weights |= {partition: weight.tolist()}
    return class_weights


def filter_out_partially_annotated(tiles_df, annotations_dir, tile_properties, paths):
    tiles_full_df = tiles_df[tiles_df['partially_annotated'].eq(False)]
    tiles_full = pd2csv(tiles_full_df, annotations_dir, f'tiles_fully_annotated_{tile_properties}')
    paths |= {'tiles_fully_annotated': tiles_full}

    tiles_partial_df = tiles_df[tiles_df['partially_annotated'].eq(True)]
    tiles_partial = pd2csv(tiles_partial_df, annotations_dir, f'tiles_partially_annotated_{tile_properties}')
    paths |= {'tiles_partially_annotated': tiles_partial}

    return tiles_full_df, tiles_partial_df, paths


def tile_split(tiles_full_df, seed, paths, class_weights,
               annotations_dir, tile_properties, len_classes, cross_validation):
    fold_df, test_df = train_test_split(
        tiles_full_df, test_size=.2, random_state=seed, shuffle=True, stratify=tiles_full_df['majority_class'])
    fold = pd2csv(fold_df, annotations_dir, f'fold_{tile_properties}')
    test = pd2csv(test_df, annotations_dir, f'test_{tile_properties}')
    paths |= {'fold': fold, 'test': test}
    class_weights = compute_class_weights(fold_df, len_classes, 'fold', class_weights)

    train_df, val_df = train_test_split(
        tiles_full_df, test_size=.1, random_state=seed, shuffle=True, stratify=None)
    train = pd2csv(train_df, annotations_dir, f'train_{tile_properties}')
    val = pd2csv(val_df, annotations_dir, f'val_{tile_properties}')
    paths |= {'train': train, 'val': val}
    class_weights = compute_class_weights(train_df, len_classes, 'train', class_weights)

    if cross_validation:
        skf = StratifiedKFold(n_splits=cross_validation, shuffle=True, random_state=seed)
        for i, (train_indices, val_indices) in enumerate(skf.split(X=fold_df, y=fold_df['majority_class'])):
            train_df = fold_df.iloc[train_indices]
            train = pd2csv(train_df, annotations_dir, f'train_{i}_{tile_properties}')
            val_df = fold_df.iloc[val_indices]
            val = pd2csv(val_df, annotations_dir, f'val_{i}_{tile_properties}')
            paths |= {f'train_{i}': train, f'val_{i}': val}
            class_weights = compute_class_weights(train_df, len_classes, f'train_{i}', class_weights)

    return paths, class_weights


def main(args):
    tile_properties = f'{args.pixel_size}_{args.tile_size}_{args.overlap}'

    tiles = f'tiles_{tile_properties}'
    annotations_dir = f'{args.splits_dir}/{tile_properties}'
    tiles_path = os.path.join(annotations_dir, f'{tiles}.csv')

    if os.path.isfile(tiles_path):
        tiles_df = pd.read_csv(tiles_path)
    else:
        tile_dir = os.path.expanduser(f"{args.dataset_dir}/tiles_{tile_properties}_{'_'.join(args.qupath_class_names)}")
        actual_class_names = ['Background'] + list(args.qupath_class_names)
        dataset_metadata = os.path.expanduser(args.dataset_metadata)
        dataset_annotation = tile_annotator(tile_dir, actual_class_names, dataset_metadata)
        tiles_df = pd.DataFrame(dataset_annotation)
        tiles = pd2csv(tiles_df, annotations_dir, f'tiles_{tile_properties}')


    paths = {}
    paths |= {'tiles': tiles}
    tiles_full_df, tiles_partial_df, paths = filter_out_partially_annotated(
        tiles_df, annotations_dir, tile_properties, paths
    )

    class_weights = {}
    paths, class_weights = tile_split(
        tiles_full_df, args.seed, paths, class_weights,
        annotations_dir, tile_properties, len(args.qupath_class_names), args.cross_validation,
    )

    os.makedirs(args.logs, exist_ok=True)

    class_weights_file = os.path.join(args.logs, 'class_weights.json')
    with open(class_weights_file, 'w') as stream:
        json.dump(class_weights, stream, indent=2)

    paths_file = os.path.join(args.logs, 'dataset_paths.json')
    with open(paths_file, 'w') as stream:
        json.dump(paths, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pixel-size',
        default=0.25,
        type=float,
        help='target output resolution in micrometer/pixel (default: 0.25)'
    )
    parser.add_argument(
        '--tile-size',
        default=512,
        type=int,
        help='original input tile size (default: 512)',
    )
    parser.add_argument(
        '--overlap',
        default=0,
        type=int,
        help='original input tile size (default: 0)',
    )

    def tuple_of_strings(arg):
        return tuple(arg.split(','))

    parser.add_argument(
        '--qupath-class-names',
        default='HP,NIFTP,PTC,Other',
        type=tuple_of_strings,
        help='QuPath class names used at annotation time (default: HP,NIFTP,PTC,Other)',
    )
    parser.add_argument(
        '--dataset-dir',
        default='~/datasets/thyroid',
        type=str,
        help='folder to load thyroid data'
    )
    parser.add_argument(
        '--dataset-metadata',
        default='~/datasets/thyroid/thyroid_dataset_metadata.csv',
        type=str,
        help='details of the WSIs like the acquisition center'
    )
    parser.add_argument(
        '--splits-dir',
        default='splits',
        type=str,
        help='folder to store the dataset splits'
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='seed to build and split the dataset'
    )
    parser.add_argument(
        '--cross-validation',
        default=5,
        type=int,
        help='whether to perform k-fold cross-validation (default: 5)',
    )
    parser.add_argument(
        '--logs',
        default='logs',
        type=str,
        help='folder to save logs'
    )
    arguments = parser.parse_args()
    main(arguments)
