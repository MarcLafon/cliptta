from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel

Kwargs: TypeAlias = Dict[str, Any]


class CALIP(AbstractOpenSetTTAModel):
    """Tent adapts a model via parameter-free cross-attention during testing.

    Once caliped, a model adapts its logits by updating once.
    """

    def __init__(
        self,
        beta_calip: List[float] = [2.0, 0.1],
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.beta_calip = beta_calip

    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        """Using a parameter-free cross-attention mechanism to refine the logits"""
        total_features = self.get_features(images)[0]
        img_global_feat = total_features[:, 0, :]
        img_spatial_feat = total_features[:, 1:, :]
        img_spatial_feat = img_spatial_feat.permute(0, 2, 1)
        logits1, logits2 = self.new_logits(img_spatial_feat, img_global_feat)
        clip_logits = self.get_logits([img_global_feat])
        logits = self.logit_scale * clip_logits[0] + logits1 * self.beta_calip[0] + logits2 * self.beta_calip[1]
        scores = self.get_scores([logits])

        return [logits], scores

    def new_logits(
            self,
            img_spatial_feat: Tensor,
            img_global_feat: Tensor,
    ) -> Tuple[List[Tensor], Tensor]:
        with torch.no_grad():
            logits1 = []
            logits2 = []
            for i, feat_v in enumerate(img_spatial_feat):
                A_weight = torch.matmul(feat_v.permute(1, 0), self.class_prototypes.t()) * 2
                A_weight1 = F.softmax(A_weight, dim=0)
                A_weight2 = F.softmax(A_weight, dim=1)

                feat_t_a = torch.matmul(feat_v, A_weight1)
                feat_v_a = torch.matmul(A_weight2, self.class_prototypes)
                feat_v_a = feat_v_a.mean(0) + feat_v_a.max(0)[0]

                l1 = 100. * img_global_feat[i] @ feat_t_a
                l2 = 100. * feat_v_a @ self.class_prototypes.t()
                logits1.append(l1.unsqueeze(0))
                logits2.append(l2.unsqueeze(0))
            logits1 = torch.cat(logits1, dim=0)
            logits2 = torch.cat(logits2, dim=0)

        return logits1, logits2
