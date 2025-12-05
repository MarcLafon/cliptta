# flake8: noqa
from torchvision import transforms


def get_zero_transforms(n_pixels: int = 224):
    zero_transforms = [
        transforms.RandomResizedCrop(n_pixels),
        transforms.RandomHorizontalFlip(),
    ]

    return transforms.Compose(zero_transforms)
