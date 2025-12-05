# flake8: noqa
import torch.utils.data as data
import numpy as np
from PIL import Image


class CIFAR_New(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, version="v6"):
        self.data = np.load("%s/cifar10.1_%s_data.npy" % (root, version))
        self.targets = np.load("%s/cifar10.1_%s_labels.npy" % (root, version)).astype(
            "long"
        )
        self.shift_type = 'original'
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.labels = self.targets
        self.classe_to_idx = {nm: i for i, nm in enumerate(self.classes)}
        self.idx_to_class = {i: nm for i, nm in enumerate(self.classes)}
        self.class_names = self.classes

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": img,
            "target": target,
            "path": "",
            "index": index,
            "name": self.idx_to_class[target],
        }

    def __len__(self):
        return len(self.targets)
