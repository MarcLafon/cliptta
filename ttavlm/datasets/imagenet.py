from typing import Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS

from ttavlm.datasets.tools.wnid_to_name import wnid_to_name


def is_valid_file(x: str) -> bool:
    return has_file_allowed_extension(x, IMG_EXTENSIONS)  # type: ignore[arg-type]


class ImagenetDataset(ImageFolder):
    def __init__(
        self, root: str, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(root, *args, is_valid_file=is_valid_file, **kwargs)
        self.shift_type = "original"
        self.labels = self.targets
        self.wnid_to_idx = self.class_to_idx
        self.idx_to_wnid = {v: k for k, v in self.wnid_to_idx.items()}
        self.class_names = [wnid_to_name(wnid) for wnid in self.classes]
        self.wnids = [self.idx_to_wnid[i] for i in self.labels]

        self.label_names = [wnid_to_name(self.idx_to_wnid[tgt]) for tgt in self.targets]

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, _ = super().__getitem__(index)
        target = self.targets[index]
        return {
            "image": img,
            "target": target,
            "path": self.imgs[index][0],
            "index": index,
            "name": self.label_names[index],
            "wnid": self.wnids[index],
        }


if __name__ == "__main__":
    dts = ImagenetDataset(
        "/share/DEEPLEARNING/datasets/"
    )
    import ipdb

    ipdb.set_trace()
