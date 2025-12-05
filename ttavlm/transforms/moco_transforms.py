from typing import Callable, Tuple, List

import torch
from torch import Tensor
from torchvision import transforms
from PIL import ImageFilter, Image
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform: Callable) -> None:
        self.base_transform = base_transform

    def __call__(self, x: Tensor) -> List[Tensor]:
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class NCropsTransform:
    def __init__(self, transform_list: List[Callable]) -> None:
        self.transform_list = transform_list

    def __call__(self, x: Tensor) -> List[Tensor]:
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma: List[float] = [0.1, 2.0]) -> None:
        self.sigma = sigma

    def __call__(self, x: Tensor) -> Tensor:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        transform_to_pil = transforms.ToPILImage()
        transform_to_tensor = transforms.ToTensor()
        blurred_images = []
        device = x.device
        for i in range(x.shape[0]):
            pil_image = transform_to_pil(x[i])
            blurred_x = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
            blurred_images.append(transform_to_tensor(blurred_x))
        x = torch.stack(blurred_images).to(device)
        # x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_moco_augmentation(aug_type: str, normalize: bool = None) -> Callable:
    if not normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    if aug_type == "moco-v2":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), antialias=None),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                normalize,
            ]
        )

    elif aug_type == "moco-v1":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), antialias=None),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=None),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC, antialias=None),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=None),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                normalize,
            ]
        )
    return None


def wqk_transforms(aug_type: str = 'moco-v2') -> Tuple[Callable, Callable, Callable]:
    w_transforms = get_moco_augmentation(aug_type='plain')
    q_transforms = get_moco_augmentation(aug_type=aug_type)
    k_transforms = get_moco_augmentation(aug_type=aug_type)

    return w_transforms, q_transforms, k_transforms
