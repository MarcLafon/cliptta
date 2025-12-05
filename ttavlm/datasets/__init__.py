from typing import Callable, Tuple, Optional, List
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from ttavlm.datasets.utils import get_template
from ttavlm.datasets.caltech101 import Caltech101Dataset
from ttavlm.datasets.cifar_new import CIFAR_New
from ttavlm.datasets.cifar10 import CIFAR10Dataset
from ttavlm.datasets.cifar10c import CIFAR10CDataset
from ttavlm.datasets.cifar100 import CIFAR100Dataset
from ttavlm.datasets.cifar100c import CIFAR100CDataset
from ttavlm.datasets.dtd import DTDDataset
from ttavlm.datasets.eurosat import EuroSATDataset
from ttavlm.datasets.fgvc_aircraft import FGVCAircraftDataset
from ttavlm.datasets.flowers102 import Flowers102Dataset
from ttavlm.datasets.food101 import Food101Dataset
from ttavlm.datasets.imagenet import ImagenetDataset
from ttavlm.datasets.imagenet_a import ImagenetADataset
from ttavlm.datasets.imagenetc import ImagenetCDataset
from ttavlm.datasets.imagenet_r import ImagenetRDataset
from ttavlm.datasets.imagenet_sketch import ImagenetSketchDataset
from ttavlm.datasets.imagenetv2 import ImagenetV2Dataset
from ttavlm.datasets.ninco import NINCODataset
from ttavlm.datasets.nincoc import NINCOCDataset
from ttavlm.datasets.officehome import OfficeHomeDataset
from ttavlm.datasets.oxford_pets import OxfordPetsDataset
from ttavlm.datasets.pacs import PACSDataset
from ttavlm.datasets.places import PlacesDataset
from ttavlm.datasets.placesc import PlacesCDataset
from ttavlm.datasets.sun397 import SUN397Dataset
from ttavlm.datasets.svhn import SVHNDataset
from ttavlm.datasets.svhnc import SVHNCDataset
from ttavlm.datasets.stanford_cars import StanfordCarsDataset
from ttavlm.datasets.ucf101 import UCF101Dataset
from ttavlm.datasets.visda import VisdaDataset

from ttavlm.datasets.image_list import ImageList
from ttavlm.datasets.image_folder import ImageFolder
from ttavlm.datasets.tools.wnid_to_name import wnid_to_name


__all__ = [
    "Caltech101Dataset",
    "CIFAR_New",
    "CIFAR10Dataset",
    "CIFAR10CDataset",
    "CIFAR100Dataset",
    "CIFAR100CDataset",
    "DTDDataset",
    "EuroSATDataset",
    "FGVCAircraftDataset",
    "Flowers102Dataset",
    "Food101Dataset",
    "ImageFolder",
    "ImagenetDataset",
    "ImagenetADataset",
    "ImagenetCDataset",
    "ImagenetRDataset",
    "ImagenetSketchDataset",
    "ImagenetV2Dataset",
    "NINCODataset",
    "NINCOCDataset",
    "OfficeHomeDataset",
    "OxfordPetsDataset",
    "PACSDataset",
    "PlacesCDataset",
    "SUN397Dataset",
    "SVHNDataset",
    "SVHNCDataset",
    "StanfordCarsDataset",
    "VisdaDataset",
    "UCF101Dataset",
    "ImageList",
    "wnid_to_name",
    "CORRUPTIONS",
    "CLEAN_DATASETS",
    "DATASET_SUITE",
    "DOMAINS",
    "get_template",
]

CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]

PACS_DOMAINS = [
    "art_painting",
    "cartoon",
    "photo",
    "sketch",
]

OFFICEHOME_DOMAINS = [
    "Art",
    "Clipart",
    "Product",
    "Real World",
]

VISDA_DOMAINS = [
    "train",
    "validation",
]

CLEAN_DATASETS = {
    "cifar10": "cifar10",
    "cifar10c": "cifar10",
    "cifar10new": "cifar10new",
    "cifar100": "cifar100",
    "cifar100c": "cifar100",
    "imagenet": "imagenet",
    "imagenetc": "imagenet",
    "visda": "visda",
    "pacs": "pacs",
    "placesc": "places",
    "officehome": "officehome",
    "imagenet-a": "imagenet-a",
    "imagenet-r": "imagenet-r",
    "imagenet-s": "imagenet-s",
    "imagenet-v2": "imagenet-v2",
    "cars": "cars",
    "caltech": "caltech",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "aircraft": "aircraft",
    "flowers": "flowers",
    "food": "food",
    "pets": "pets",
    "sun": "sun",
    "ucf": "ucf",
}

DATASET_SUITE = [
    "cars",
    "caltech",
    "dtd",
    "eurosat",
    "aircraft",
    "flowers",
    "food",
    "pets",
    "sun",
    "ucf",
]

DOMAINS = [
    "imagenet",
    "imagenet-a",
    "imagenet-r",
    "imagenet-s",
    "imagenet-v2",
]


def _convert_image_to_rgb(image: Image) -> Image:
    return image.convert("RGB")


def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # clip mean
                std=[0.26862954, 0.26130258, 0.27577711],  # clip std
            ),
        ],
    )


def return_train_val_datasets(
    name: str,
    data_dir: str,
    train_transform: Callable[[Image.Image], Tensor],
    val_transform: Callable[[Image.Image], Tensor],
    shift: Optional[str] = None,
    severity: Optional[int] = None,
    download: Optional[bool] = False,
) -> Tuple[Dataset, Dataset, List[str]]:
    if name == "imagenet":
        train_dataset = None
        val_dataset = ImagenetDataset(
            root=os.path.join(data_dir, "imagenet", "ILSVRC", "Data", "CLS-LOC", "val"),
            transform=val_transform,
        )
    elif name == "imagenet-a":
        train_dataset = None

        val_dataset = ImagenetADataset(
            root=os.path.join(data_dir, "domains"),
            transform=val_transform,
        )
    elif name == "imagenet-r":
        train_dataset = None

        val_dataset = ImagenetRDataset(
            root=os.path.join(data_dir, "domains"),
            transform=val_transform,
        )
    elif name == "imagenet-s":
        train_dataset = None

        val_dataset = ImagenetSketchDataset(
            root=os.path.join(data_dir, "domains"),
            transform=val_transform,
        )
    elif name == "imagenet-v2":
        train_dataset = None

        val_dataset = ImagenetV2Dataset(
            root=os.path.join(data_dir, "domains"),
            transform=val_transform,
        )
    elif name == "imagenetc":
        train_dataset = None
        val_dataset = ImagenetCDataset(
            root=os.path.join(data_dir, "Imagenet-C"),
            shift_type=shift,
            severity=severity,
            transform=val_transform,
        )
    elif name == "cifar10":
        train_dataset = CIFAR10Dataset(
            root=os.path.join(data_dir, "CIFAR-10"),
            train=True,
            transform=train_transform,
        )
        val_dataset = CIFAR10Dataset(
            root=os.path.join(data_dir, "CIFAR-10"),
            train=False,
            transform=val_transform,
        )
    elif name == "cifar100":
        train_dataset = CIFAR100Dataset(
            root=os.path.join(data_dir, "CIFAR-100"),
            train=True,
            transform=train_transform,
        )
        val_dataset = CIFAR100Dataset(
            root=os.path.join(data_dir, "CIFAR-100"),
            train=False,
            transform=val_transform,
        )
    elif name == "cifar10c":
        train_dataset = None
        val_dataset = CIFAR10CDataset(
            root=os.path.join(data_dir, "CIFAR-10"),
            corruption_root=os.path.join(data_dir, "CIFAR-10-C"),
            shift_type=shift,
            severity=severity,
            transform=val_transform,
        )
    elif name == "cifar10new":
        train_dataset = None
        val_dataset = CIFAR_New(
            root=os.path.join(data_dir, "CIFAR-10.1"),
            transform=val_transform,
        )
    elif name == "cifar100c":
        train_dataset = None
        val_dataset = CIFAR100CDataset(
            root=os.path.join(data_dir, "CIFAR-100"),
            corruption_root=os.path.join(data_dir, "CIFAR-100-C"),
            shift_type=shift,
            severity=severity,
            transform=val_transform,
        )
    elif name == "visda":
        train_dataset = None
        if shift is None:
            shift = "train"
        val_dataset = VisdaDataset(
            root=os.path.join(data_dir, "visda", shift),
            domain=shift,
            transform=val_transform,
        )
    elif name == "pacs":
        train_dataset = None
        if shift is None:
            shift = "photo"
        val_dataset = PACSDataset(
            root=os.path.join(data_dir, "pacs", shift),
            domain=shift,
            transform=val_transform,
        )
    elif name == "officehome":
        train_dataset = None
        if shift is None:
            shift = "Art"
        val_dataset = OfficeHomeDataset(
            root=os.path.join(data_dir, "officehome", shift),
            domain=shift,
            transform=val_transform,
        )
    elif name == "cars":
        train_dataset = StanfordCarsDataset(
            root=os.path.join(data_dir, "dataset_suite", "stanford_cars"),
            split="train",
            transform=train_transform,
        )
        val_dataset = StanfordCarsDataset(
            root=os.path.join(data_dir, "dataset_suite", "stanford_cars"),
            split="test",
            transform=val_transform,
        )
    elif name == "caltech":
        train_dataset = Caltech101Dataset(
            root=os.path.join(data_dir, "dataset_suite", "caltech101"),
            split="train",
            transform=train_transform,
        )
        val_dataset = Caltech101Dataset(
            root=os.path.join(data_dir, "dataset_suite", "caltech101"),
            split="test",
            transform=val_transform,
        )
    elif name == "dtd":
        train_dataset = DTDDataset(
            root=os.path.join(data_dir, "dataset_suite", "dtddataset"),
            split="train",
            transform=train_transform,
        )
        val_dataset = DTDDataset(
            root=os.path.join(data_dir, "dataset_suite", "dtddataset"),
            split="test",
            transform=val_transform,
        )
    elif name == "eurosat":
        train_dataset = EuroSATDataset(
            root=os.path.join(data_dir, "dataset_suite", "EuroSAT_RGB"),
            split="train",
            transform=train_transform,
        )
        val_dataset = EuroSATDataset(
            root=os.path.join(data_dir, "dataset_suite", "EuroSAT_RGB"),
            split="test",
            transform=val_transform,
        )
    elif name == "aircraft":
        train_dataset = FGVCAircraftDataset(
            root=os.path.join(data_dir, "dataset_suite"),
            split="train",
            transform=train_transform,
        )
        val_dataset = FGVCAircraftDataset(
            root=os.path.join(data_dir, "dataset_suite"),
            split="test",
            transform=val_transform,
        )
    elif name == "flowers":
        train_dataset = Flowers102Dataset(
            root=os.path.join(data_dir, "dataset_suite", "flowers102"),
            split="train",
            transform=train_transform,
        )
        val_dataset = Flowers102Dataset(
            root=os.path.join(data_dir, "dataset_suite", "flowers102"),
            split="test",
            transform=val_transform,
        )
    elif name == "food":
        train_dataset = Food101Dataset(
            root=os.path.join(data_dir, "dataset_suite", "food-101"),
            split="train",
            transform=train_transform,
        )
        val_dataset = Food101Dataset(
            root=os.path.join(data_dir, "dataset_suite", "food-101"),
            split="test",
            transform=val_transform,
        )
    elif name == "pets":
        train_dataset = OxfordPetsDataset(
            root=os.path.join(data_dir, "dataset_suite", "oxford-iiit-pet"),
            split="train",
            transform=train_transform,
        )
        val_dataset = OxfordPetsDataset(
            root=os.path.join(data_dir, "dataset_suite", "oxford-iiit-pet"),
            split="test",
            transform=val_transform,
        )
    elif name == "sun":
        train_dataset = SUN397Dataset(
            root=os.path.join(data_dir, "dataset_suite", "SUN397"),
            split="train",
            transform=train_transform,
        )
        val_dataset = SUN397Dataset(
            root=os.path.join(data_dir, "dataset_suite", "SUN397"),
            split="test",
            transform=val_transform,
        )
    elif name == "ucf":
        train_dataset = UCF101Dataset(
            root=os.path.join(data_dir, "dataset_suite", "UCF-101-midframes"),
            split="train",
            transform=train_transform,
        )
        val_dataset = UCF101Dataset(
            root=os.path.join(data_dir, "dataset_suite", "UCF-101-midframes"),
            split="test",
            transform=val_transform,
        )
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    return train_dataset, val_dataset


def return_ood_dataset(
    ood_dataset_name: str,
    data_dir: str,
    shift_type: str,
    severity: int,
    transform: Optional[Callable],
) -> Dataset:
    if ood_dataset_name == "svhn":
        return SVHNDataset(
            root=os.path.join(data_dir, "SVHN"),
            transform=transform,
        )
    if ood_dataset_name == "places":
        return PlacesDataset(
            root=os.path.join(data_dir, "ood_data", "Places"),
            transform=transform,
        )
    elif ood_dataset_name == "placesc":
        return PlacesCDataset(
            root=os.path.join(data_dir, "ood_data", "Places-C"),
            shift_type=shift_type,
            severity=severity,
            transform=transform,
        )
    elif ood_dataset_name == "svhnc":
        return SVHNCDataset(
            root=os.path.join(data_dir, "SVHN"),
            corruption_root=os.path.join(data_dir, "SVHN-C"),
            shift_type=shift_type,
            severity=severity,
            transform=transform,
        )
    if ood_dataset_name == "places":
        return ImageFolder(
            root=os.path.join(data_dir, "ood_data", "Places"),
            transform=transform,
        )
    elif ood_dataset_name == "placesc":
        return PlacesCDataset(
            root=os.path.join(data_dir, "ood_data", "Places-C"),
            shift_type=shift_type,
            severity=severity,
            transform=transform,
        )
    elif ood_dataset_name == "ninco":
        return NINCODataset(
            root=os.path.join(data_dir, "ood_data", "NINCO"),
            transform=transform,
        )
    elif ood_dataset_name == "nincoc":
        return NINCOCDataset(
            root=os.path.join(data_dir, "ood_data", "NINCO-C"),
            shift_type=shift_type,
            severity=severity,
            transform=transform,
        )
    else:
        raise NotImplementedError
