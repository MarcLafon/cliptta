from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch

from torch import Tensor
from functools import partial

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.memory import HUS
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class SoTTA(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        capacity: int,
        high_threshold: float,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        num_classes = len(self.class_prototypes)
        self.memory = HUS(capacity, num_classes=num_classes, threshold=high_threshold)

    @torch.enable_grad()
    def _forward_and_adapt(
            self,
            images: List[Tensor],
    ) -> Tensor:
        features = self.get_features(images)
        logits = self.get_logits(features)
        loss_tta = lib.softmax_entropy(self.logit_scale * logits[0]).mean(0)
        loss_tta.backward()

        return loss_tta

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

        if step == 0:
            # Adding to memory
            image_features = self.get_features(images)
            logits = self.get_logits(image_features)
            scores = self.get_scores(logits, score_type='max_prob')
            output = logits[0].view(-1, len(self.class_prototypes))
            pseudo_classes = output.max(1, keepdim=False)[1]
            for i in range(len(logits[0])):
                image = images[0][i].unsqueeze(0).detach().cpu()
                pseudo_cls = pseudo_classes[i].cpu().numpy().item()
                pseudo_conf = scores[i].detach().cpu().numpy()
                self.memory.add_instance([image, pseudo_cls, pseudo_conf])

        # Sample and adapt
        image_mem, _ = self.memory.get_memory()
        if len(image_mem) != 0:
            image_mem = [im.to(device) for im in image_mem]

            _ = self._forward_and_adapt(image_mem)

            closure = partial(self._forward_and_adapt, images=image_mem) if self.use_sam else None

            self.optimizer.step(closure)
            self.optimizer.zero_grad(set_to_none=True)

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None

        return logits, scores

    def _reset_extra(self) -> None:
        self.memory.reset()
