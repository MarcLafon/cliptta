from typing import Dict, List, Any
import os

from ttavlm.datasets import ImagenetDataset


class ImagenetCDataset(ImagenetDataset):
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


if __name__ == "__main__":
    dts = ImagenetCDataset(
        root="/share/DEEPLEARNING/datasets/Imagenet-C",
        shift_type="gaussian_noise",
        severity=5,
    )
    import ipdb

    ipdb.set_trace()
