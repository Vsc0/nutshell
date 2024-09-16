import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchmetrics import classification, Dice

from unet import UNet
from augmenter import custom_transforms
from generator import ThyroidImageDataset as Dataset


def main(args, verbose=True):
    if verbose:
        print('torch.__version__', torch.__version__)
        print('torch.backends.cudnn.version()', torch.backends.cudnn.version())
        print('torch.backends.cudnn.is_available()', torch.backends.cudnn.is_available())

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'

    args_file = os.path.join(args.logs, 'predictor_args.json')
    with open(args_file, 'w') as stream:
        json.dump(vars(args), stream)

    paths_file = os.path.join(args.logs, 'dataset_paths.json')
    with open(paths_file, 'r') as stream:
        paths = json.load(stream)

    transform = custom_transforms(
        resize_size=args.image_size,
    )
    target_transform = None

    class_names = args.qupath_class_names
    num_classes = len(class_names)
    actual_num_classes = len(class_names) + 1  # also take into account the "Background" class

    Dataset.out_channels = actual_num_classes

    if args.discard_last_class:  # discard the "Other" class in metrics computation
        num_classes = actual_num_classes - 1

    none_kwargs = {
        'num_classes': num_classes,
        'top_k': 1,
        'average': 'none',
        'multidim_average': 'global',
        'ignore_index': None,
        'validate_args': False,
    }
    none_p = classification.MulticlassPrecision(**none_kwargs).to(device)
    none_r = classification.MulticlassRecall(**none_kwargs).to(device)
    none_f1s = classification.MulticlassF1Score(**none_kwargs).to(device)
    none_ji = classification.MulticlassJaccardIndex(
        num_classes=num_classes, average='none', ignore_index=None, validate_args=False).to(device)

    weighted_kwargs = {
        'num_classes': num_classes,
        'top_k': 1,
        'average': 'weighted',
        'multidim_average': 'global',
        'ignore_index': 0,
        'validate_args': False,
    }
    weighted_p = classification.MulticlassPrecision(**weighted_kwargs).to(device)
    weighted_r = classification.MulticlassRecall(**weighted_kwargs).to(device)
    weighted_f1s = classification.MulticlassF1Score(**weighted_kwargs).to(device)
    weighted_ji = classification.MulticlassJaccardIndex(
        num_classes=num_classes, average='weighted', ignore_index=0, validate_args=False).to(device)
    d = Dice(num_classes=num_classes, ignore_index=0, top_k=1).to(device)

    ds = Dataset(
        annotations=paths[args.partition],
        transform=transform,
        target_transform=target_transform,
    )
    dl = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    with torch.set_grad_enabled(False):
        model = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels, init_features=32)
        state_dict = torch.load(f'{args.weights}/model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        metrics = {
            'cm': [],
            'p': [], 'mcp': [],
            'r': [], 'mcr': [],
            'f1s': [], 'mcf1s': [],
            'ji': [], 'mcji': [],
            'd': [],
        }

        for j, (x, y) in tqdm(enumerate(dl)):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            if args.discard_last_class:
                y_pred = y_pred[:, 0:4, :, :]
                y = torch.where(y == 4, 0, y)

            none_p(y_pred, y)
            none_r(y_pred, y)
            none_f1s(y_pred, y)
            none_ji(y_pred, y)
            weighted_p(y_pred, y)
            weighted_r(y_pred, y)
            weighted_f1s(y_pred, y)
            weighted_ji(y_pred, y)
            d(y_pred, y)

        mcp = weighted_p.compute().tolist()
        weighted_p.reset()
        print(f'Precision: {round(mcp, 7)}')
        metrics['p'].append(mcp)
        mcp = none_p.compute().tolist()
        none_p.reset()
        print(f'Precision: {[round(x, 7) for x in mcp]}')
        metrics['mcp'].append(mcp)

        mcr = weighted_r.compute().tolist()
        weighted_r.reset()
        print(f'Recall: {round(mcr, 7)}')
        metrics['r'].append(mcr)
        mcr = none_r.compute().tolist()
        none_r.reset()
        print(f'Recall: {[round(x, 7) for x in mcr]}')
        metrics['mcr'].append(mcr)

        mcf1s = weighted_f1s.compute().tolist()
        weighted_f1s.reset()
        print(f'F1-Score: {round(mcf1s, 7)}')
        metrics['f1s'].append(mcf1s)
        mcf1s = none_f1s.compute().tolist()
        none_f1s.reset()
        print(f'F1-Score: {[round(x, 7) for x in mcf1s]}')
        metrics['mcf1s'].append(mcf1s)

        mcji = weighted_ji.compute().tolist()
        weighted_ji.reset()
        print(f'Jaccard Index: {round(mcji, 7)}')
        metrics['ji'].append(mcji)
        mcji = none_ji.compute().tolist()
        none_ji.reset()
        print(f'Jaccard Index: {[round(x, 7) for x in mcji]}')
        metrics['mcji'].append(mcji)

        mcd = d.compute().tolist()
        d.reset()
        print(f'Dice: {round(mcd, 7)}')
        metrics['d'].append(mcd)

        metrics_file = os.path.join(args.logs, f'{args.partition}_metrics.json')
        with open(metrics_file, 'w') as stream:
            json.dump(metrics, stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Infer thyroid resection (HE-stained) cell nuclei segmentation and classification'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help='device for training (default: cuda:0)',
    )
    parser.add_argument(
        '--logs',
        default='logs',
        type=str,
        help='folder to save logs'
    )
    parser.add_argument(
        '--batch-size',
        default=8,
        type=int,
        help='input batch size for training (default: 8)',
    )
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='number of workers for data loading (default: 4)',
    )
    parser.add_argument(
        '--weights',
        default='weights',
        type=str,
        help='folder to save weights'
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='seed the run model inference (default: 42)'
    )
    parser.add_argument(
        '--image-size',
        default=512,
        type=int,
        help='target input image size (default: 512)',
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='test',
        help='partition to predict on (default: test)',
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
        '--discard-last-class',
        action='store_false',
        help='whether to discard the "Other" class in metrics computation (default: True)',
    )
    arguments = parser.parse_args()
    main(arguments, verbose=True)
