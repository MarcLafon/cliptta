from typing import Dict, List, Any, Union
import os

from torch import Tensor
from torchvision.datasets import ImageFolder


class PlacesCDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        shift_type: str,
        severity: int,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        root = os.path.join(root, shift_type, str(severity))
        super().__init__(root, *args, **kwargs)

        self.shift_type = shift_type
        self.severity = severity

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "name": self.classes[target],
        }


if __name__ == "__main__":
    dts = PlacesCDataset(
        root="/share/DEEPLEARNING/datasets/",
        shift_type="gaussian_noise",
        severity=5,
    )
    import ipdb

    ipdb.set_trace()
