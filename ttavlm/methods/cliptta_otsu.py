from typing import Dict, Any, Optional, Tuple, List
from typing_extensions import TypeAlias

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.models.clip import tokenize as clip_tokenize
from ttavlm.memory import CCM


import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class CLIPTTA(AbstractOpenSetTTAModel):
    """
    CLIPTTA adapts CLIP using the same loss as during the pre-training.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        use_memory: bool = False,
        use_softmax_entropy: bool = False,
        use_scheduler: bool = False,
        use_tent: bool = False,
        use_clipartt: bool = False,
        K: int = 3,
        clipartt_temp: Optional[float] = 0.01,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.template = template
        self.class_names = class_names
        self.use_memory = use_memory
        self.use_softmax_entropy = use_softmax_entropy
        self.use_scheduler = use_scheduler
        self.use_tent = use_tent
        self.use_clipartt = use_clipartt
        self.K = K
        self.clipartt_temp = clipartt_temp
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        self.memory = None
        if use_memory:
            self.memory = CCM(num_shots=self.num_shots, num_classes=len(self.class_names), sample_size=self.sample_size)

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        if self.measure_improvement:
            self.model0 = deepcopy(self.model)
            for param in self.model0.parameters():
                param.detach()

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
    ) -> Tensor:
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)

        # Regularization loss
        if self.beta_reg != 0:
            logits = self.get_logits(image_features)
            loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])
        else:
            loss_reg = 0.0

        # OOD loss computation
        if self.use_ood_loss or self.detect_ood:
            scores = self.get_scores(logits, image_features[0])
            if self.use_ood_loss:
                loss_ood = self.get_otsu_loss(scores)
            else:
                loss_ood = 0.0
        else:
            loss_ood = 0.0

        # OOD-ID separation
        if self.detect_ood:
            images, image_features = self.filter_id(images, image_features, scores, self.alpha if self.update_alpha else None)

        # TTA loss computation
        if self.update_text:
            class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=True)
        else:
            class_prototypes = self.class_prototypes

        # Compute TTA loss using samples from the batch
        loss_tta = self.compute_loss_tta(image_features, class_prototypes)

        # Compute TTA loss using samples from the memory
        if self.use_memory:
            # Updating Memory
            if step == 0:
                with torch.no_grad():
                    logits = self.get_logits(image_features)
                    _, pred = logits[0].topk(1, 1, True, True)
                    scores = self.get_scores(logits, image_features, score_type=self.id_score_type)

                self.memory.update(images[0].cpu().detach(), pred[:, 0].cpu().detach(), scores.cpu().detach())

            # Sampling from memory
            images, _, _ = self.memory.sample()
            image_features = self.get_features([images.to(image_features[0].device)])
            logits = self.get_logits(image_features)
            loss_tta += self.compute_loss_tta(image_features, class_prototypes)
            loss_reg += lib.softmax_mean_entropy(self.logit_scale * logits[0])

        # Final loss
        loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg + self.beta_ood * loss_ood

        loss.backward()
        return loss

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        _ = self._forward_and_adapt(images, step)

        closure = partial(self._forward_and_adapt, images=images, step=step) if self.use_sam else None

        self.optimizer.step(closure)
        self.optimizer.zero_grad(set_to_none=True)

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                if self.update_text:
                    class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder)
                else:
                    class_prototypes = self.class_prototypes
                image_features = self.get_features(images)
                logits = self.get_logits(image_features, class_prototypes)
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None

        return logits, scores

    def compute_loss_tta(self, image_features: List[Tensor], class_prototypes: Tensor) -> Tensor:
        # Compute pseudo-labels
        logits = image_features[0] @ class_prototypes.t()
        _, pred = logits.topk(1, 1, True, True)
        pred_text_features = class_prototypes[pred[:, 0]]

        # Compute logits (image v.s. pseudo-captions, i.e. size is B x B)
        logits_per_image = self.logit_scale * image_features[0] @ pred_text_features.t()
        logits_per_text = logits_per_image.t() if self.update_text else logits_per_image

        # TTA loss
        if self.use_tent:
            loss_tta = lib.softmax_entropy(self.logit_scale * logits).mean(0)
        elif self.use_clipartt:
            _, pred = logits.topk(self.K, 1, True, True)
            if self.K == 1:
                text_features = self.class_prototypes[pred[:, 0]]
            else:
                text_prompts = lib.getprompt(self.K, pred.cpu().numpy(), self.class_names, self.template[0])
                pred_inputs = clip_tokenize(text_prompts).to(logits.device)

                # With the new prompts, compute the image-to-image and text-to-text similarities to get targets
                with torch.no_grad():
                    text_features = self.clip_text_encoder(pred_inputs)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

            images_similarity = image_features[0] @ image_features[0].t()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.clipartt_temp, dim=-1)

            # Obtain new logits (image v.s. new prompt, i.e. size is B x B)
            predictions = (self.logit_scale * text_features @ image_features[0].t()).t()
            loss_tta = F.cross_entropy(predictions, targets)
        else:
            if self.use_softmax_entropy:
                loss_tta = (lib.softmax_entropy(logits_per_image).mean(0) + lib.softmax_entropy(logits_per_text).mean(0)) / 2
            else:
                targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
                loss_tta = (self.loss_fn(logits_per_image, targets).mean(0) + self.loss_fn(logits_per_text, targets).mean(0)) / 2

        return loss_tta

    def _reset_extra(self) -> None:
        if self.use_memory:
            self.memory.reset()

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        if self.use_scheduler:
            self.scheduler.step()
