from typing import Optional, Tuple, Callable
from os.path import join

import timm

import torch
import torch.nn as nn
from torch import device
from torchvision import transforms

from ttavlm.models.clip import load as load_clip
from ttavlm.lib import LOGGER

from ttavlm.models.clip import available_models, load, tokenize, CLIP


__all__ = [
    "available_models",
    "load",
    "tokenize",
    "CLIP",
    "return_base_model",
]


WEIGHTS = {
    "cifar10": {
        "resnet50": "resnet50_cifar10.pth",
    },
    "cifar10c": {
        "resnet50": "resnet50_cifar10.pth",
    },
}

classes = {
    "cifar10": 10,
    "cifar10c": 10,
    "cifar100": 100,
    "cifar100c": 100,
    "visda": 12,
    "imagenet": 1000,
    "imagenetc": 1000,
}


def return_base_model(
    name: str,
    device: device,
    dataset: str,
    path_to_weights: str = None,
    segments: Optional[int] = 0,
) -> Tuple[nn.Module, Callable]:
    if name.startswith("resnet"):
        LOGGER.info(f"Loading Resnet version: {name}")
        model = timm.create_model(
            model_name=name,
            pretrained=dataset == "imagenet",
            num_classes=classes[dataset],
        )
        if dataset in ["cifar10", "cifar10c", "cifar100", "cifar100c"]:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Loading weights
        if path_to_weights is not None:
            weights = torch.load(join(path_to_weights, WEIGHTS[dataset][name]))["state_dict"]
            model.load_state_dict(weights, strict=False)

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        model.to(device)
        model.dtype = model.fc.weight.dtype

    elif name.startswith("clip"):
        LOGGER.info(f"Loading CLIP version: {name[5:]}")
        model, val_transform = load_clip(name[5:], device=device, segments=segments)
        model.visual.dtype = model.visual.conv1.weight.dtype
    else:
        raise NotImplementedError

    return model, val_transform
