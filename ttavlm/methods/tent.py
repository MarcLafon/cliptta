from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
from torch import Tensor
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class Tent(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if self.measure_improvement:
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
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)

        # TTA loss (entropy minimization)
        loss_tta = lib.softmax_entropy(self.logit_scale * logits[0]).mean(0)

        # Regularization loss
        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])

        # Final loss
        loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg
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

    def compute_loss(self, image_features: Tensor) -> Tensor:
        logits = self.get_logits([image_features])

        # TTA loss (entropy minimization)
        loss_tta = lib.softmax_entropy(self.logit_scale * logits[0]).mean(0)
        # Regularization loss
        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])

        # Final loss
        loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg
        return loss


class TentOracle(AbstractOpenSetTTAModel):
    """
    This class is used for exeperiments on Tent with oracle information such
      as missclassified and/or OOD examples
    """

    def __init__(
        self,
        oracle_miss: bool = True,
        oracle_ood: bool = True,
        miss_weight: float = -0.1,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.oracle_miss = oracle_miss
        self.oracle_ood = oracle_ood
        self.miss_weight = miss_weight

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
        n_id = labels.shape[0]
        n_ood = images.shape[0] - n_id

        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        scores = self.get_scores(logits)

        w = torch.ones_like(scores)

        if self.oracle_miss:
            w[:n_id][labels != logits[0][:n_id].argmax(dim=1)] = self.miss_weight

        if not self.closed_set:
            if self.oracle_ood:
                w[n_ood:] = 0.0

        loss_tta = (w * lib.softmax_entropy(self.logit_scale * logits[0])).mean(0)
        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])
        loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, scores
