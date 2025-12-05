from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.transforms import get_tta_transforms, wqk_transforms

Kwargs: TypeAlias = Dict[str, Any]


class AdaContrast(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        beta_ins: float = 0.1,
        aug_type: str = 'moco-v2',
        queue_size: int = 16384,
        n_neighbors: int = 3,
        m: float = 0.999,
        T_moco: float = 0.07,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.beta_ins = beta_ins
        self.aug_type = aug_type
        self.transforms = get_tta_transforms(n_pixels=224)
        self.momentum_model = deepcopy(self.model)
        self.momentum_model.requires_grad_(False)
        self.queue_size = queue_size
        self.n_neighbors = n_neighbors
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # Creating memory bank
        self.register_buffer("mem_feat", torch.randn(self.model.output_dim, queue_size).cuda())
        self.register_buffer("mem_labels", torch.randint(0, len(self.class_prototypes), (queue_size,)).cuda())

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
        # Computing features bank
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        probs = logits[0].softmax(dim=1)
        rand_idxs = torch.randperm(len(image_features[0])).to(logits[0].device)
        banks = {
            "features": image_features[0][rand_idxs][:self.queue_size],
            "probs": probs[rand_idxs][:self.queue_size],
            "prt": 0,
        }

        # Transforming images
        w_transform, q_transform, k_transform = wqk_transforms(self.aug_type)
        images_w = [w_transform(images[0])]
        images_q = [q_transform(images[0])]
        images_k = [k_transform(images[0])]

        feats_w = self.get_features(images_w, self.model)
        logits_w = self.get_logits(feats_w)
        probs_w = logits_w[0].softmax(dim=1)

        feats_q = self.get_features(images_q, self.model)
        logits_q = self.get_logits(feats_q)
        probs_q = logits_q[0].softmax(dim=1)

        with torch.no_grad():
            feats_k = self.get_features(images_k, self.momentum_model)  # keys

        l_pos = torch.einsum("nc,nc->n", [feats_q[0], feats_k[0]]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [feats_q[0].float(), self.mem_feat.clone().detach().float().to(probs_q.device)])
        logits_ins = torch.cat([l_pos, l_neg], dim=1) / self.T_moco

        # Getting pseudolabels from weak transformations
        with torch.no_grad():
            pseudo_labels_w, probs_w = self.refine_predictions(feats_w[0], probs_w, banks)

        # Update memory
        self.update_memory(feats_k[0], pseudo_labels_w)

        # Instance loss
        labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).to(logits_ins.device)
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels_w.reshape(-1, 1) != self.mem_labels
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(logits_ins.device))
        loss_ins = F.cross_entropy(logits_ins, labels_ins)

        # Classification loss
        loss_cls = F.cross_entropy(logits_q[0], pseudo_labels_w)

        # Diversification loss
        probs_w_mean = probs_w.mean(dim=0)
        loss_w_div = -torch.sum(-probs_w_mean * torch.log(probs_w_mean + 1e-8))
        probs_q_mean = probs_q.mean(dim=0)
        loss_q_div = -torch.sum(-probs_q_mean * torch.log(probs_q_mean + 1e-8))
        loss_div = loss_w_div + loss_q_div

        # Final loss
        loss = self.beta_tta * loss_cls + self.beta_ins * loss_ins + self.beta_reg * loss_div
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

    @torch.no_grad()
    def update_memory(self, keys: Tensor, pseudo_labels: Tensor) -> None:
        # keys = self.concat_all_gather(keys)
        # pseudo_labels = self.concat_all_gather(pseudo_labels)

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.queue_size
        self.mem_feat[:, idxs_replace] = keys.float().T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.queue_size

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
                self.model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def refine_predictions(self,
                           features: Tensor,
                           probs: Tensor,
                           banks: Dict,
                           method: str = "nearest_neighbors",
                           ) -> Tuple[Tensor, Tensor]:
        features_bank = banks["features"]
        probs_bank = banks["probs"]
        if method == "nearest_neighbors":
            pred_labels, probs = self.soft_k_nearest_neighbors(features, features_bank, probs_bank)
        else:
            pred_labels = probs.argmax(dim=1)

        return pred_labels, probs

    @torch.no_grad()
    def soft_k_nearest_neighbors(self,
                                 features: Tensor,
                                 features_bank: Tensor,
                                 probs_bank: Tensor,
                                 distance_type: str = 'euclidean'
                                 ) -> Tuple[Tensor, Tensor]:
        pred_probs = []
        for feats in features.split(64):
            distances = self.get_distances(feats, features_bank, distance_type)
            _, idxs = distances.sort()
            idxs = idxs[:, :self.n_neighbors]
            # (64, num_nbrs, num_classes), average over dim=1
            probs = probs_bank[idxs, :].mean(1)
            pred_probs.append(probs)
        pred_probs = torch.cat(pred_probs)
        _, pred_labels = pred_probs.max(dim=1)

        return pred_labels, pred_probs

    def get_distances(self,
                      X: Tensor,
                      Y: Tensor,
                      dist_type: str = "euclidean"
                      ) -> Tensor:
        """
        Args:
            X: (N, D) tensor
            Y: (M, D) tensor
        """
        if dist_type == "euclidean":
            distances = torch.cdist(X.float(), Y.float())
        elif dist_type == "cosine":
            distances = 1 - torch.matmul(X, Y.T)
            # distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
        else:
            raise NotImplementedError(f"{dist_type} distance not implemented.")

        return distances
