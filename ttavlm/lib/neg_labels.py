negative_classes_cifar = [
    "object",
    "thing",
    "stuff",
    "wall",
    "glass",
    "blob",
]

negative_classes_imagenet = []

negative_classes = {
    "cifar10": negative_classes_cifar,
    "cifar100": negative_classes_cifar,
    "imagenet": negative_classes_imagenet,
}
