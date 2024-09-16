import random
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


def custom_transforms(
        random_horizontal_flip_prob: float = None,
        random_vertical_flip_prob: float = None,
        axis_aligned_rotation: bool = False,
        resize_size: int = None,
):

    transform_list = []
    if random_horizontal_flip_prob is not None:
        transform_list.append(T.RandomHorizontalFlip(p=random_horizontal_flip_prob))
    if random_vertical_flip_prob is not None:
        transform_list.append(T.RandomVerticalFlip(p=random_vertical_flip_prob))
    if axis_aligned_rotation is not False:
        transform_list.append(RandomRotationTransform(angles=(0, 90, 180, 270)))
    if resize_size is not None:
        transform_list.append(ResizeTransform(size=resize_size))
    return T.Compose(transform_list)


class RandomRotationTransform(nn.Module):
    def __init__(self, angles):
        super().__init__()
        self.angles = angles

    def forward(self, image, target):
        angle = random.choice(self.angles)
        image = TF.rotate(image, angle)
        target = TF.rotate(target, angle)
        return image, target


class ResizeTransform(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image, target):
        image = TF.resize(image, [self.size], antialias=True)
        target = TF.resize(target, [self.size], interpolation=T.InterpolationMode.NEAREST, antialias=True)
        return image, target
