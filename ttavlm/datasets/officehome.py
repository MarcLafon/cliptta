from typing import Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import ImageFolder


def reformat_name(name: str) -> str:
    atoms = name.split('_')
    class_name = ' '.join(atoms)

    return class_name


class OfficeHomeDataset(ImageFolder):

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
            "name": reformat_name(self.idx_to_class[target]),
        }


if __name__ == "__main__":
    dts = OfficeHomeDataset(
        root="/export/datasets/",
    )
