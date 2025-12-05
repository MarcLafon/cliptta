# flake8: noqa
from typing import List, Tuple

import numpy as np
import torch

from ttavlm.models.clip import clip

def getprompt_old(K, c, classes, template="a photo of a {}"):
    '''
    What is K, what is c ?
    '''
    new_class = classes[c[0]]
    for k in range(1, K):
        new_class = new_class + " or " + classes[c[k]]
    return template.format(new_class)

def getprompt(K, pred, classes, template="a photo of a {}"):
    classes = np.char.array(classes)
    pred_classes = classes[pred[:, 0]]
    for k in range(1, K):
        pred_classes = pred_classes + " or " + classes[pred[:, k]]

    text_prompt = list(map(lambda x: template.format(x), pred_classes))
    return text_prompt

def get_text_features(
        classnames: List[str],
        templates: List[str],
        clip_model: torch.nn.Module,
        enable_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    clip_weights = []

    if enable_grad:
        templates = [np.random.choice(templates)]

    for template in templates:
        with torch.no_grad():
            texts = [template.format(c.replace('_', ' ')) for c in classnames]
            texts = clip.tokenize(texts).cuda()
            with torch.set_grad_enabled(enable_grad):
                text_features = clip_model.encode_text(texts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                clip_weights.append(text_features)

    clip_weights = torch.stack(clip_weights, dim=0).mean(0)
    clip_weights = clip_weights / clip_weights.norm(dim=-1, keepdim=True)
    class_bias = torch.zeros(len(classnames), device=texts.device)    
    return clip_weights, class_bias
