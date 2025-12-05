from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import math
import torch
import operator
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class TDA(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        pos_alpha_beta: List[float] = [2.0, 5.0],
        neg_alpha_beta: List[float] = [0.117, 1.0],
        pos_shot_capacity: int = 3,
        neg_shot_capacity: int = 2,
        entropy_threshold: List[float] = [0.2, 0.5],
        mask_threshold: List[float] = [0.03, 1.0],
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        entropy_dict = {'lower': entropy_threshold[0],
                        'upper': entropy_threshold[1]}
        mask_dict = {'lower': mask_threshold[0],
                     'upper': mask_threshold[1]}
        self.pos_params = {'shot_capacity': pos_shot_capacity,
                           'alpha': pos_alpha_beta[0],
                           'beta': pos_alpha_beta[1]}
        self.neg_params = {'shot_capacity': neg_shot_capacity,
                           'alpha': neg_alpha_beta[0],
                           'beta': neg_alpha_beta[1],
                           'entropy_threshold': entropy_dict,
                           'mask_threshold': mask_dict}
        self.pos_cache = {}
        self.neg_cache = {}
        self.accuracies = []

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
        logits = self.logit_scale * self.get_logits(image_features)[0]
        image_features, logits, loss, prob_map, pred = self.process_features(logits, image_features[0])
        prop_entropy = self.get_entropy(loss)
        self.update_cache(self.pos_cache, pred, [image_features, loss], self.pos_params['shot_capacity'])
        if self.neg_params['entropy_threshold']['lower'] < prop_entropy < self.neg_params['entropy_threshold']['upper']:
            self.update_cache(self.neg_cache, pred, [image_features, loss, prob_map], self.neg_params['shot_capacity'], True)

        final_logits = logits.clone()
        if self.pos_cache:
            final_logits += self.compute_cache_logits(image_features, self.pos_cache, self.pos_params)
        if self.neg_cache:
            lower_mask_threshold = self.neg_params['mask_threshold']['lower']
            upper_mask_threshold = self.neg_params['mask_threshold']['upper']
            final_logits -= self.compute_cache_logits(image_features, self.neg_cache, self.neg_params, [lower_mask_threshold, upper_mask_threshold])
        scores = self.get_scores([final_logits])

        return [final_logits], scores

    def process_features(
            self,
            logits: Tensor,
            image_features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        if image_features.size(0) > 1:
            batch_entropy = lib.softmax_entropy(logits)
            m = max(1, int(batch_entropy.size()[0] * 0.1))
            selected_idx = torch.argsort(batch_entropy, descending=False)[:m]
            output = logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            loss = self.avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = lib.softmax_entropy(logits)
            prob_map = logits.softmax(1)
            pred = int(logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, logits, loss, prob_map, pred

    def update_cache(
            self,
            cache: Dict,
            pred: int,
            features_loss: List[Tensor],
            shot_capacity: int,
            include_prob_map: bool = False,
    ) -> None:
        with torch.no_grad():
            item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
            if pred in cache:
                if len(cache[pred]) < shot_capacity:
                    cache[pred].append(item)
                elif features_loss[1] < cache[pred][-1][1]:
                    cache[pred][-1] = item
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            else:
                cache[pred] = [item]

    def compute_cache_logits(
            self,
            image_features: Tensor,
            cache: Dict,
            params: Dict,
            neg_mask_thresholds: List[float] = None,
    ) -> Tensor:
        """Compute logits using positive/negative cache."""
        with torch.no_grad():
            cache_keys = []
            cache_values = []
            for class_index in sorted(cache.keys()):
                for item in cache[class_index]:
                    cache_keys.append(item[0])
                    if neg_mask_thresholds:
                        cache_values.append(item[2])
                    else:
                        cache_values.append(class_index)
            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
            if neg_mask_thresholds:
                cache_values = torch.cat(cache_values, dim=0)
                cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
            else:
                cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=self.class_prototypes.t().size(1))).cuda().half()

            affinity = image_features @ cache_keys
            cache_logits = ((-1) * (params['beta'] - params['beta'] * affinity)).exp() @ cache_values

            return params['alpha'] * cache_logits

    def get_entropy(
            self,
            loss: Tensor
    ) -> float:
        max_entropy = math.log2(self.class_prototypes.t().size(1))

        return float(loss / max_entropy)

    def avg_entropy(
            self,
            outputs: Tensor,
    ) -> Tensor:
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
