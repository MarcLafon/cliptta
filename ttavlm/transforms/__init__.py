from typing import Callable, List

from torchvision.transforms import Compose

from ttavlm.transforms.transform_list import TransformList
from ttavlm.transforms.tta_transforms import get_tta_transforms, get_simple_tta_transforms
from ttavlm.transforms.zero_transforms import get_zero_transforms
from ttavlm.transforms.moco_transforms import get_moco_augmentation, wqk_transforms

__all__ = [
    "TransformList",
    "get_tta_transforms",
    "get_simple_tta_transforms",
    "get_zero_transforms",
    "get_moco_augmentation",
    "wqk_transforms",
    "add_tta_transform",
]


def add_tta_transform(val_transform: Callable, n_pixels: int, style: str = "simple") -> List[Callable]:
    """
    Assuming that val_transform ends with [..., ToTensor(), Normalize()]
    """
    if style == "simple":
        tta_transform = get_simple_tta_transforms(n_pixels)
    elif style == "zero":
        tta_transform = get_zero_transforms(n_pixels)
    else:
        tta_transform = get_tta_transforms(n_pixels)

    return Compose(tta_transform.transforms + val_transform.transforms[-2:])
    # return Compose(val_transform.transforms[:-2] + tta_transform.transforms + val_transform.transforms[-2:])
