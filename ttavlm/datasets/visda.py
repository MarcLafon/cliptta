# flake8: noqa
from typing import Callable, Optional, List, Dict, Any, Union
# TODO: Refactor this to follow the same logic as in other datasets
import os

from PIL import Image
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class VisdaDataset(ImageFolder):
    def __init__(
        self, root: str, domain: str, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(root, *args, **kwargs)
        self.shift_type = domain
        self.labels = self.targets
        self.idx_to_class = {idx: c for (c, idx) in self.class_to_idx.items()}
        self.class_names = self.classes

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, _ = super().__getitem__(index)
        target = self.targets[index]
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "name": self.idx_to_class[target],
        }
