from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
from torch import Tensor
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class OSTTA(AbstractOpenSetTTAModel):
    """
    OSTTA adapts CLIP with entropy on filtered samples.
    """

    def __init__(
        self,
        margin: float = 0.05,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.margin = margin
        self.model0 = deepcopy(self.model)
        for param in self.model0.parameters():
            param.detach()

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        with torch.no_grad():
            image_features0 = self.get_features(images, self.model0)
            logits0 = self.get_logits(image_features0)
            prob0 = (self.logit_scale * logits0[0]).softmax(1)
            values0, indices0 = prob0.max(1)

        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        prob = (self.logit_scale * logits[0]).softmax(1)
        values = prob[torch.arange(prob0.size(0)), indices0]

        # TTA loss (filtered entropy minimization)
        entropys = lib.softmax_entropy(self.logit_scale * logits[0])
        filter_ids = values - values0 >= self.margin
        if len(entropys[filter_ids]) > 0:
            loss_tta = entropys[filter_ids].mean(0)
        else:
            loss_tta = entropys.mean(0)

        # Regularization loss
        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])

        # Final loss
        loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Get the final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None

        return logits, scores
