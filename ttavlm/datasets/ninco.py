from typing import Dict, List, Any, Union
import os

from torch import Tensor
from torchvision.datasets import ImageFolder


class NINCODataset(ImageFolder):
    folder_dir = "NINCO_OOD_classes"

    def __init__(
        self, root: str, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(os.path.join(root, self.folder_dir), *args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "name": self.classes[target],
        }
