# flake8: noqa
from typing import List, Optional


templates = [
    "a photo of a {}",
    "itap of a {}",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

negative_classes = [
    "object",
    "thing",
    "stuff",
    "wall",
    "glass",
    "blob",
]

all_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

def get_template(
        name: str,
        template_type: Optional[str] = "default",
) -> List[str]:
    assert template_type in ["default", "select",
                             "all"], f"unkown value for template_type ({template_type}). Please use one of ['default', 'select', 'all']."
    if name == "imagenet":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "imagenet-a":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "imagenet-r":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "imagenet-s":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "imagenet-v2":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "imagenetc":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "cifar10":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "cifar100":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "cifar10c":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "cifar10new":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "cifar100c":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "visda":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "pacs":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "officehome":
        if template_type == "default":
            template = ["a photo of a {}"]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "cars":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "caltech":
        if template_type == "default":
            template = ["A photo of a {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "dtd":
        if template_type == "default":
            template = ["{} texture."]
        elif template_type == "select":
            template = [t.replace("{}", "{} texture") for t in templates]
        else:
            template = [t.replace("{}", "{} texture") for t in all_templates]
    elif name == "eurosat":
        if template_type == "default":
            template = ["A centered satellite photo of {}."]
        elif template_type == "select":
            template = [t.replace("{}", "centered satellite {}") for t in templates]
        else:
            template = [t.replace("{}", "centered satellite {}") for t in all_templates]
    elif name == "aircraft":
        if template_type == "default":
            template = ["A photo of a {}, a type of aircraft."]
        elif template_type == "select":
            template = [t.replace("{}", "{}, a type of aircraft") for t in templates]
        else:
            template = [t.replace("{}", "{}, a type of aircraft") for t in all_templates]
    elif name == "flowers":
        if template_type == "default":
            template = ["A photo of a {}, a type of flower."]
        elif template_type == "select":
            template = [t.replace("{}", "{}, a type of flower") for t in templates]
        else:
            template = [t.replace("{}", "{}, a type of flower") for t in all_templates]
    elif name == "food":
        if template_type == "default":
            template = ["A photo of a {}, a type of food."]
        elif template_type == "select":
            template = [t.replace("{}", "{}, a type of food") for t in templates]
        else:
            template = [t.replace("{}", "{}, a type of food") for t in all_templates]
    elif name == "pets":
        if template_type == "default":
            template = ["A photo of a {}, a type of pet."]
        elif template_type == "select":
            template = [t.replace("{}", "{}, a type of pet") for t in templates]
        else:
            template = [t.replace("{}", "{}, a type of pet") for t in all_templates]
    elif name == "sun":
        if template_type == "default":
            template = ["A photo of {}."]
        elif template_type == "select":
            template = templates
        else:
            template = all_templates
    elif name == "ucf":
        if template_type == "default":
            template = ["A photo of a person doing {}."]
        elif template_type == "select":
            template = [t.replace("{}", "person doing {}") for t in templates]
        else:
            template = [t.replace("{}", "person doing {}") for t in all_templates]
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    return template
