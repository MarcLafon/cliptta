from typing import Callable, Any, Dict, Union
from pathlib import Path
import os
from PIL import Image
import numpy as np

from torch import Tensor
from torchvision.datasets import SVHN


class SVHNCDataset(SVHN):
    def __init__(
        self,
        root: Union[str, Path],
        corruption_root: Union[str, Path],
        shift_type: str,
        severity: int,
        split: str = "test",
        transform: Union[Callable[..., Any], None] = None,
        target_transform: Union[Callable[..., Any], None] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.corruption_root = corruption_root
        self.shift_type = shift_type
        self.severity = severity

        self.targets = self.labels
        corrupted_data = np.load(
            os.path.join(self.corruption_root, f"{self.shift_type}.npy")
        )
        corrupted_data = corrupted_data[
            (self.severity - 1) * len(self.labels): self.severity * len(self.labels)
        ]
        self.data = corrupted_data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int, None]]:
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(img)

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
