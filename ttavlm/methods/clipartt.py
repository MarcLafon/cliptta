from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.models.clip import tokenize as clip_tokenize
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class CLIPArTT(AbstractOpenSetTTAModel):
    """
    CLIPArTT adapts CLIP using pseudo-labels constructed from image-image similarity and text-text similarity
    matrices.
    """

    def __init__(
        self,
        class_names: List[str],
        template: List[str] = None,
        temp: float = 0.01,
        K: int = 3,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_names = class_names
        self.template = template
        self.K = K
        self.temp = temp
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        device = images[0].device
        # Getting the top-K most probable classes from the initial prediction
        image_features = self.get_features(images)
        similarity = self.get_logits(image_features)[0]
        values, pred = similarity.topk(self.K, 1, True, True)
        if self.K == 1:
            text_features = self.class_prototypes[pred[:, 0]]
        else:
            text_prompts = lib.getprompt(self.K, pred.cpu().numpy(), self.class_names, self.template[0])
            pred_inputs = clip_tokenize(text_prompts).to(device)
            # With the new prompts, compute the image-to-image and text-to-text similarities to get targets
            with torch.no_grad():
                text_features = self.clip_text_encoder(pred_inputs)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

        images_similarity = image_features[0] @ image_features[0].t()
        texts_similarity = text_features @ text_features.t()
        targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.temp, dim=-1)

        # Obtain new logits (image v.s. new prompt, i.e. size is B x B)
        predictions = (self.logit_scale * text_features @ image_features[0].t()).t()

        # Using crossentropy between targets and logits
        loss = self.loss_fn(predictions, targets.detach())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features([images[0].to(device)])
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits)
        else:
            logits, scores = None, None

        return logits, scores


class CLIPArTTOracle(AbstractOpenSetTTAModel):
    """
    This class is used for experiments on CLIPArTT with oracle information such
    as missclassified and/or OOD examples
    """

    def __init__(
        self,
        class_names: List[str],
        temp: float = 0.01,
        K: int = 3,
        oracle_miss: bool = True,
        oracle_ood: bool = True,
        miss_weight: float = -0.1,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.K = K
        self.class_names = class_names
        self.temp = temp

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.oracle_miss = oracle_miss
        self.oracle_ood = oracle_ood
        self.miss_weight = miss_weight

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
        **kwargs: Kwargs,
    ) -> Tuple[List[Tensor], Tensor]:
        n_id = labels.shape[0]
        n_ood = images[0].shape[0] - n_id

        device = images[0].device

        # Getting the top-K most probable classes from the initial prediction
        image_features = self.get_features(images)
        similarity = self.get_logits(image_features)[0]
        values, pred = similarity.topk(self.K, 1, True, True)
        if self.K == 1:
            text_features = self.class_prototypes[pred[:, 0]]
        else:
            pred_inputs = torch.cat([clip_tokenize(lib.getprompt(self.K, c, self.class_names, self.template[0])) for c in pred]).to(device)

            # With the new prompts, compute the image-to-image and text-to-text similarities to get targets
            with torch.no_grad():
                text_features = self.clip_text_encoder(pred_inputs)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

        images_similarity = image_features[0] @ image_features[0].t()
        texts_similarity = text_features @ text_features.t()
        targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.temp, dim=-1)

        # Obtain new prediction (image v.s. new prompt, i.e. size is B x B)
        predictions = (self.logit_scale * text_features @ image_features[0].t()).t()

        # Weights for OOD samples and ID missclassification
        w = torch.ones_like(pred[:, 0])
        if self.oracle_miss:
            w[:n_id][labels != similarity[:n_id].argmax(dim=1)] = self.miss_weight

        if not self.closed_set:
            if self.oracle_ood:
                w[n_ood:] = 0.0

        loss = (w * self.loss_fn(predictions, targets)).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits)
        else:
            logits, scores = None, None

        return logits, scores
