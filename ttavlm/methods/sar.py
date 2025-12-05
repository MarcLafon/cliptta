from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import math
import torch
import numpy as np

from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class SAR(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        alpha_entropy: float = 0.4,
        reset_constant_em: float = 0.2,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        n_classes = len(self.class_prototypes)
        self.margin_e0 = alpha_entropy * math.log(n_classes)
        self.reset_constant_em = reset_constant_em
        self.ema = None
        self.reset_flag = False

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

        # First loss backward
        entropys = lib.softmax_entropy(self.logit_scale * logits[0])
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss_first = entropys.mean()
        loss_first.backward()
        self.optimizer.first_step(zero_grad=True)

        # EMA update
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        entropys2 = lib.softmax_entropy(self.logit_scale * logits[0])
        entropys2 = entropys2[filter_ids_1]
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = self.update_ema(loss_second.item())

        # Second loss backward
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        # Model recovery
        if self.ema is not None:
            if self.ema < 0.2:
                self.reset()

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits)
        else:
            logits, scores = None, None

        return logits, scores

    def update_ema(self, new_data: float) -> float:
        if self.ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * self.ema + (1 - 0.9) * new_data
