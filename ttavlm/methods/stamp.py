from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import math
import random
from copy import deepcopy
from functools import partial

import torch
from torch import Tensor

from ttavlm.methods import AbstractOpenSetTTAModel
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class RBM:
    def __init__(
        self,
        max_len: int,
        num_class: int,
    ) -> None:
        self.num_class = num_class
        self.count_class = torch.zeros(num_class)
        self.data = [[] for _ in range(num_class)]
        self.max_len = max_len
        self.total_num = 0

    def remove_item(self) -> None:
        max_count = 0
        for i in range(self.num_class):
            if len(self.data[i]) == 0:
                continue
            if self.count_class[i] > max_count:
                max_count = self.count_class[i]
        max_classes = []
        for i in range(self.num_class):
            if self.count_class[i] == max_count and len(self.data[i]) > 0:
                max_classes.append(i)
        remove_class = random.choice(max_classes)
        self.data[remove_class].pop(0)

    def append(self, items: Tensor, class_ids: Tensor) -> None:
        for item, class_id in zip(items, class_ids):
            if self.total_num < self.max_len:
                self.data[class_id].append(item)
                self.total_num += 1
            else:
                self.remove_item()
                self.data[class_id].append(item)

    def get_data(self) -> Tensor:
        data = []
        for cls in range(self.num_class):
            data.extend(self.data[cls])
            self.count_class[cls] = 0.9 * self.count_class[cls] + 0.1 * len(self.data[cls])
        return torch.stack(data)

    def __len__(self) -> int:
        return self.total_num

    def reset(self) -> None:
        self.count_class = torch.zeros(self.num_class)
        self.data = [[] for _ in range(self.num_class)]
        self.total_num = 0


class STAMP(AbstractOpenSetTTAModel):
    def __init__(
        self,
        memory_length: int,
        alpha_stamp: float,
        use_consistency_filtering: bool,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.memory_length = memory_length
        self.alpha_stamp = alpha_stamp
        self.use_consistency_filtering = use_consistency_filtering

        self.num_classes = self.class_prototypes.shape[0]
        self.margin = alpha_stamp * math.log(self.num_classes)

        self.memory = RBM(self.memory_length, self.num_classes)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        # Copying source model for consistency filtering
        if self.use_consistency_filtering:
            self.norm_model = deepcopy(self.model).train()

    def consistency_filtering(self, images_origin: Tensor) -> Tensor:
        image_features = self.get_features(images_origin)
        logits = self.get_logits(image_features)[0]
        preds = logits.argmax(dim=1)

        image_features_source = self.get_features(images_origin, self.norm_model)
        logits_source = self.get_logits(image_features_source)[0]
        preds_source = logits_source.argmax(dim=1)

        return preds == preds_source

    def update_memory(self, images: List[Tensor]) -> List[Tensor]:

        self.model.train()

        if self.use_consistency_filtering:
            filter_ids_0 = self.consistency_filtering(images)
        else:
            filter_ids_0 = torch.BoolTensor([True] * images[0].shape[0]).to(images[0].device)

        # Confidence_filtering
        images_features = self.get_features(images)
        logits = self.get_logits(images_features)
        probs = [(self.logit_scale * lg).softmax(dim=-1) for lg in logits]
        probs = torch.stack(probs, dim=0).mean(dim=0)
        entropy = lib.entropy(probs)
        filter_ids = (entropy < self.margin) * filter_ids_0

        # Updating memory
        self.memory.append(images[0][filter_ids], probs.argmax(dim=1)[filter_ids])

        return logits

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_features = self.get_features([images])
        logits = self.get_logits(image_features)[0]

        entropys = lib.softmax_entropy(logits)
        inv_entropy = 1 / torch.exp(entropys)
        coeff = inv_entropy / inv_entropy.sum() * self.memory.max_len
        entropys = entropys.mul(coeff)
        loss = entropys.mean()
        loss.backward()
        return loss

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        with torch.no_grad():
            logits = self.update_memory(images)  # Warning: returns List[Tensor]!

        # adapt model with updated memory
        if len(self.memory) > 0:
            data = self.memory.get_data()
            self.optimizer.zero_grad()
            _ = self._forward_and_adapt(data)
            closure = partial(self._forward_and_adapt, images=data)
            self.optimizer.step(closure if self.use_sam else None)
            self.optimizer.zero_grad(set_to_none=True)

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits)
        else:
            logits, scores = None, None

        return logits, scores

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        if len(self.memory) > 0:
            pass
            # self.scheduler.step()

    def _reset_extra(self) -> None:
        self.memory.reset()
