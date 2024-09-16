import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2.functional as TF


class ThyroidImageDataset(Dataset):
    """Thyroid map-style dataset for cell nuclei segmentation and classification"""

    in_channels = 3
    out_channels = 5

    def __init__(
            self,
            annotations: str,
            transform: nn.Sequential = None,
            target_transform: nn.Sequential = None,
    ):
        self.annotations = pd.read_csv(annotations)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]

        image = Image.open(annotation['tile_name'])
        image = tv_tensors.Image(image, dtype=torch.uint8, device='cpu', requires_grad=False)

        target = read_image(annotation['sem_seg_file_name'])
        # target_filename = annotation['sem_seg_file_name']
        # target = read_image(f'{target_filename[:-4]}_sparse{target_filename[-4:]}')
        target = tv_tensors.Mask(target, dtype=torch.uint8, device='cpu', requires_grad=False)

        if self.transform:
            image, target = self.transform(image, target)

        image = TF.to_dtype(inpt=image, dtype=torch.float32, scale=True)
        target = target.squeeze().long()

        return image, target
