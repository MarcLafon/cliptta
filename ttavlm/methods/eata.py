from typing import Dict, Any, Tuple, List, Optional
from typing_extensions import TypeAlias

import math
import torch
import torch.nn.functional as F

from torch import Tensor
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class ETA(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        d_margin: Optional[float] = 0.05,
        alpha_entropy: Optional[float] = 0.4,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model0 = deepcopy(self.model)
        for param in self.model0.parameters():
            param.detach()
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        n_classes = len(self.class_prototypes)
        self.e_margin = math.log(n_classes) * alpha_entropy
        self.d_margin = d_margin
        self.current_model_probs = None

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
        entropys = lib.softmax_entropy(self.logit_scale * logits[0])
        filter_ids_1 = torch.where(entropys < self.e_margin)
        entropys = entropys[filter_ids_1]
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), logits[0][filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = self.update_model_probs(logits[0][filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = self.update_model_probs(logits[0][filter_ids_1].softmax(1))

        self.num_samples_update_1 += entropys.size(0)
        self.num_samples_update_2 += filter_ids_1[0].size(0)
        self.reset_model_probs(updated_probs)

        # TTA filtered entropy loss
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        entropys = entropys.mul(coeff)
        loss_tta = entropys.mean(0)

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

    def update_model_probs(self, new_probs: Tensor) -> Tensor:
        if self.current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return self.current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * self.current_model_probs + (1 - 0.9) * new_probs.mean(0)

    def reset_model_probs(self, probs: Tensor) -> None:
        self.current_model_probs = probs
