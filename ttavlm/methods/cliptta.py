from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.memory import CCM

import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class CLIPTTA_Old(AbstractOpenSetTTAModel):
    """
    CLIPTTA adapts CLIP using the same loss as during the pre-training.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        use_softmax_entropy: bool = False,
        use_memory: bool = False,
        use_scheduler: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_names = class_names
        self.template = template
        self.use_softmax_entropy = use_softmax_entropy
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.use_memory = use_memory
        self.use_scheduler = use_scheduler

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        if self.use_memory:
            self.memory = CCM(num_shots=self.num_shots, num_classes=len(class_names), sample_size=max(self.sample_size, len(class_names)))

        if self.measure_improvement:
            self.model0 = deepcopy(self.model)
            for param in self.model0.parameters():
                param.detach()

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: List[Tensor],
    ) -> Tensor:
        if self.update_text:
            class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=True)
        else:
            class_prototypes = self.class_prototypes

        image_features = self.get_features(images)
        similarity = image_features[0] @ class_prototypes.t()

        _, pred = similarity.topk(1, 1, True, True)
        pred_text_features = class_prototypes[pred[:, 0]]

        # Obtain new logits (image v.s. new prompt, i.e. size is B x B)
        logits_per_image = self.logit_scale * image_features[0] @ pred_text_features.t()
        logits_per_text = logits_per_image.t() if self.update_text else logits_per_image

        if self.use_softmax_entropy:
            loss = ((lib.softmax_entropy(logits_per_image)).mean(0) + (lib.softmax_entropy(logits_per_text)).mean(0)) / 2
        else:
            targets = torch.eye(images[0].shape[0]).to(images[0].device)
            loss = ((self.loss_fn(logits_per_image, targets)).mean(0) + (self.loss_fn(logits_per_text, targets)).mean(0)) / 2

        loss -= self.beta_reg * lib.softmax_mean_entropy(self.logit_scale * similarity)
        loss.backward()
        return loss

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        if step == 0 and self.use_memory:
            with torch.no_grad():
                if self.update_text:
                    text_features, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=False)
                else:
                    text_features = self.class_prototypes
                image_features = self.get_features(images)
                logits = [image_feature @ text_features.t() for image_feature in image_features]
                pred = logits[0].topk(1, 1, True, True)[1][:, 0]
                scores = self.get_scores(logits, image_features)

            self.memory.update(images[0].cpu().detach(), pred.cpu().detach(), scores.cpu().detach())
        # import ipdb; ipdb.set_trace()
        adapt_samples = [self.memory.sample()[0].cuda(non_blocking=True)] if self.use_memory else images
        _ = self._forward_and_adapt(adapt_samples)

        closure = partial(self._forward_and_adapt, images=adapt_samples) if self.use_sam else None

        self.optimizer.step(closure)
        self.optimizer.zero_grad(set_to_none=True)

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                if self.update_text:
                    text_features, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=False)
                else:
                    text_features = self.class_prototypes
                image_features = self.get_features(images)
                logits = [image_feature @ text_features.t() for image_feature in image_features]
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None
        return logits, scores

    def compute_loss(self, image_features: Tensor) -> Tensor:
        similarity = image_features @ self.class_prototypes.t()
        _, pred = similarity.topk(1, 1, True, True)
        pred_text_features = self.class_prototypes[pred[:, 0]]
        logits_per_image = self.logit_scale * image_features @ pred_text_features.t()
        loss = (lib.softmax_entropy(logits_per_image)).mean(0)
        loss -= self.beta_reg * lib.softmax_mean_entropy(self.logit_scale * similarity)
        return loss

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        if self.use_scheduler:
            self.scheduler.step()
