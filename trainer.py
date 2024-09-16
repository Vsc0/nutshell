import os
import json
import time
import copy
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import classification, Dice
from codecarbon import track_emissions

from unet import UNet
from augmenter import custom_transforms
from generator import ThyroidImageDataset as Dataset


@track_emissions(offline=True, country_iso_code='ITA')
def main(args, verbose=True):
    if verbose:
        print('torch.__version__', torch.__version__)
        print('torch.backends.cudnn.version()', torch.backends.cudnn.version())
        print('torch.backends.cudnn.is_available()', torch.backends.cudnn.is_available())

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'

    args_file = os.path.join(args.logs, 'trainer_args.json')
    with open(args_file, 'w') as stream:
        json.dump(vars(args), stream)

    paths_file = os.path.join(args.logs, 'dataset_paths.json')
    with open(paths_file, 'r') as stream:
        paths = json.load(stream)

    weights_file = os.path.join(args.logs, 'class_weights.json')
    with open(weights_file, 'r') as stream:
        class_weights = json.load(stream)

    train_transform = custom_transforms(
        random_horizontal_flip_prob=.5,
        random_vertical_flip_prob=.5,
        axis_aligned_rotation=True,
        resize_size=args.image_size,
    )
    val_transform = custom_transforms(
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

    os.makedirs(args.weights, exist_ok=True)

    perform_cv = args.cross_validation != 0
    for i in range(1 if perform_cv else 0, args.cross_validation + 1):
        train_ds = Dataset(
            annotations=paths[f'train_{i}' if perform_cv else 'train'],
            transform=train_transform,
            target_transform=target_transform,
        )
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_ds = Dataset(
            annotations=paths[f'val_{i}' if perform_cv else 'val'],
            transform=val_transform,
            target_transform=target_transform,
        )
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
        )

        model = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels, init_features=32)
        if verbose:
            print(f'Using {model}')
        model = model.to(args.device)

        weight = torch.FloatTensor(class_weights[f'train_{i}' if perform_cv else 'train']).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')

        params = model.parameters()
        optimizer = optim.Adam(params, lr=args.lr)

        step_size = 30
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=.1)

        since = time.time()

        if args.early_stopping:
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = 7
            counter = 0
            best_val_loss = float('inf')
            min_delta = 0.0005

        metrics_per_epoch = {
            'train_p': [], 'train_mcp': [],
            'train_r': [], 'train_mcr': [],
            'train_f1s': [], 'train_mcf1s': [],
            'train_ji': [], 'train_mcji': [],
            'train_d': [],
            'train_loss': [],
            'val_p': [], 'val_mcp': [],
            'val_r': [], 'val_mcr': [],
            'val_f1s': [], 'val_mcf1s': [],
            'val_ji': [], 'val_mcji': [],
            'val_d': [],
            'val_loss': [],
        }

        for _ in tqdm(range(args.epochs), desc='Epoch', total=args.epochs):
            model.train()
            train_size = len(train_dl)
            train_running_loss = 0.0
            for _, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                train_running_loss += loss.item()
                optimizer.zero_grad(True)
                loss.backward()
                optimizer.step()

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

            partition = 'train'

            mcp = weighted_p.compute().tolist()
            weighted_p.reset()
            print(f'{partition.capitalize()} Precision: {round(mcp, 7)}')
            metrics_per_epoch[f'{partition}_p'].append(mcp)
            mcp = none_p.compute().tolist()
            none_p.reset()
            print(f'{partition.capitalize()} Precision: {[round(x, 7) for x in mcp]}')
            metrics_per_epoch[f'{partition}_mcp'].append(mcp)

            mcr = weighted_r.compute().tolist()
            weighted_r.reset()
            print(f'{partition.capitalize()} Recall: {round(mcr, 7)}')
            metrics_per_epoch[f'{partition}_r'].append(mcr)
            mcr = none_r.compute().tolist()
            none_r.reset()
            print(f'{partition.capitalize()} Recall: {[round(x, 7) for x in mcr]}')
            metrics_per_epoch[f'{partition}_mcr'].append(mcr)

            mcf1s = weighted_f1s.compute().tolist()
            weighted_f1s.reset()
            print(f'{partition.capitalize()} F1-Score: {round(mcf1s, 7)}')
            metrics_per_epoch[f'{partition}_f1s'].append(mcf1s)
            mcf1s = none_f1s.compute().tolist()
            none_f1s.reset()
            print(f'{partition.capitalize()} F1-Score: {[round(x, 7) for x in mcf1s]}')
            metrics_per_epoch[f'{partition}_mcf1s'].append(mcf1s)

            mcji = weighted_ji.compute().tolist()
            weighted_ji.reset()
            print(f'{partition.capitalize()} Jaccard Index: {round(mcji, 7)}')
            metrics_per_epoch[f'{partition}_ji'].append(mcji)
            mcji = none_ji.compute().tolist()
            none_ji.reset()
            print(f'{partition.capitalize()} Jaccard Index: {[round(x, 7) for x in mcji]}')
            metrics_per_epoch[f'{partition}_mcji'].append(mcji)

            mcd = d.compute().tolist()
            d.reset()
            print(f'{partition.capitalize()} Dice: {round(mcd, 7)}')
            metrics_per_epoch[f'{partition}_d'].append(mcd)

            train_loss = train_running_loss / train_size
            print(f'{partition.capitalize()} Loss: {train_loss:>7f}\n')
            metrics_per_epoch[f'{partition}_loss'].append(train_loss)

            model.eval()
            val_size = len(val_dl)
            val_running_loss = 0.0
            optimizer.zero_grad(True)
            with torch.no_grad():
                for _, (x, y) in enumerate(val_dl):
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    val_running_loss += loss.item()

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

            partition = 'val'

            mcp = weighted_p.compute().tolist()
            weighted_p.reset()
            print(f'{partition.capitalize()} Precision: {round(mcp, 7)}')
            metrics_per_epoch[f'{partition}_p'].append(mcp)
            mcp = none_p.compute().tolist()
            none_p.reset()
            print(f'{partition.capitalize()} Precision: {[round(x, 7) for x in mcp]}')
            metrics_per_epoch[f'{partition}_mcp'].append(mcp)

            mcr = weighted_r.compute().tolist()
            weighted_r.reset()
            print(f'{partition.capitalize()} Recall: {round(mcr, 7)}')
            metrics_per_epoch[f'{partition}_r'].append(mcr)
            mcr = none_r.compute().tolist()
            none_r.reset()
            print(f'{partition.capitalize()} Recall: {[round(x, 7) for x in mcr]}')
            metrics_per_epoch[f'{partition}_mcr'].append(mcr)

            mcf1s = weighted_f1s.compute().tolist()
            weighted_f1s.reset()
            print(f'{partition.capitalize()} F1-Score: {round(mcf1s, 7)}')
            metrics_per_epoch[f'{partition}_f1s'].append(mcf1s)
            mcf1s = none_f1s.compute().tolist()
            none_f1s.reset()
            print(f'{partition.capitalize()} F1-Score: {[round(x, 7) for x in mcf1s]}')
            metrics_per_epoch[f'{partition}_mcf1s'].append(mcf1s)

            mcji = weighted_ji.compute().tolist()
            weighted_ji.reset()
            print(f'{partition.capitalize()} Jaccard Index: {round(mcji, 7)}')
            metrics_per_epoch[f'{partition}_ji'].append(mcji)
            mcji = none_ji.compute().tolist()
            none_ji.reset()
            print(f'{partition.capitalize()} Jaccard Index: {[round(x, 7) for x in mcji]}')
            metrics_per_epoch[f'{partition}_mcji'].append(mcji)

            mcd = d.compute().tolist()
            d.reset()
            print(f'{partition.capitalize()} Dice: {round(mcd, 7)}')
            metrics_per_epoch[f'{partition}_d'].append(mcd)

            val_loss = val_running_loss / val_size
            print(f'{partition.capitalize()} Loss: {val_loss:>7f}\n')
            metrics_per_epoch[f'{partition}_loss'].append(val_loss)

            if args.early_stopping:
                if val_loss <= best_val_loss - min_delta:
                    counter = 0
                    print(f'Best Val Loss: {best_val_loss:>7f}\nVal Epoch Loss: {val_loss:>7f}\n')
                    best_val_loss = val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    counter += 1
                    print(f'{counter} Epochs with No Improvement!')
                    if counter == patience:
                        print('Early Stopping!')
                        break

            scheduler.step()
            print('\n')

        time_elapsed = time.time() - since
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if args.early_stopping:
            print(f'Min Val Epoch Loss: {best_val_loss:>7f}')
            model.load_state_dict(best_model_weights)

        model = model.to('cpu')
        torch.save(
            model.state_dict(), os.path.join(args.weights, f'model_fold_{i}.pt' if perform_cv else 'model.pt')
        )
        example = torch.rand(1, Dataset.in_channels, args.image_size, args.image_size)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(
            os.path.join(args.weights, f'traced_model_fold_{i}.pt' if perform_cv else 'traced_model.pt')
        )
        metrics_file = os.path.join(
            args.logs, f'metrics_per_epoch_fold_{i}.json' if perform_cv else 'metrics_per_epoch.json'
        )
        with open(metrics_file, 'w') as stream:
            json.dump(metrics_per_epoch, stream)
        print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a U-Net model for segmentation and classification of thyroid resection (HE-stained) cell nuclei'
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
        default=8,
        type=int,
        help='number of workers for data loading (default: 8)',
    )
    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help='initial learning rate (default: 0.0001)',
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--cross-validation',
        default=0,
        type=int,
        help='whether to perform k-fold cross-validation (default: 0)',
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='whether to apply the early stopping regularization technique to avoid overfitting (default: False)',
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
        help='seed train the model (default: 42)'
    )
    parser.add_argument(
        '--image-size',
        default=512,
        type=int,
        help='target input image size (default: 512)',
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
