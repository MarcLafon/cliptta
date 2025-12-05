# flake8: noqa

from typing import Callable, Any, Dict, Union
from pathlib import Path
import os

import numpy as np
from torch import Tensor
from torchvision.datasets import CIFAR100


class CIFAR100CDataset(CIFAR100):
    def __init__(
        self,
        root: Union[str, Path],
        corruption_root: Union[str, Path],
        shift_type: str,
        severity: int,
        transform: Union[Callable[..., Any], None] = None,
        target_transform: Union[Callable[..., Any], None] = None,
    ) -> None:
        super().__init__(root, False, transform, target_transform, False)
        self.corruption_root = corruption_root
        self.shift_type = shift_type
        self.severity = severity

        self.class_names = self.classes
        self.labels = self.targets
        self.idx_to_class = {idx: c for (c, idx) in self.class_to_idx.items()}
        tesize = len(self.targets)
        teset_raw = np.load(
            os.path.join(self.corruption_root, f"{self.shift_type}.npy")
        )
        teset_raw = teset_raw[(self.severity - 1) * tesize : self.severity * tesize]
        self.data = teset_raw

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(
        self, index: int
    ) -> Dict[
        str,
        Union[
            Tensor,
            str,
            int,
        ],
    ]:
        img, target = super().__getitem__(index)

        return {
            "image": img,
            "target": target,
            "path": "",
            "index": index,
            "name": self.idx_to_class[target],
        }


if __name__ == "__main__":
    dts = CIFAR100CDataset(
        root="/share/DEEPLEARNING/datasets/CIFAR-100",
        corruption_root="/share/DEEPLEARNING/datasets/CIFAR-100-C",
        shift_type="gaussian_noise",
        severity=5,
    )
    import ipdb

    ipdb.set_trace()
