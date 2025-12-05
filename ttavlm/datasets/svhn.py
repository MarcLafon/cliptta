from typing import Callable, Any, Dict, Union

from pathlib import Path
from PIL import Image
import numpy as np

from torch import Tensor
from torchvision.datasets import SVHN


class SVHNDataset(SVHN):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "test",
        transform: Union[Callable[..., Any], None] = None,
        target_transform: Union[Callable[..., Any], None] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.targets = self.labels

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int, None]]:
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": img,
            "target": target,
            "path": "",
            "index": index,
            "name": str(target),
        }
