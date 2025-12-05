from typing import Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import ImageFolder


class PlacesDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(root, *args, **kwargs)

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
    dts = PlacesDataset(
        root="/share/DEEPLEARNING/datasets/",
        shift_type="gaussian_noise",
        severity=5,
    )
    import ipdb

    ipdb.set_trace()
