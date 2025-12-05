from typing import Dict, Any, Optional, Tuple, List
from typing_extensions import TypeAlias

import torch
from torch import Tensor
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.models.clip import tokenize as clip_tokenize
from ttavlm.memory import CCM
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class UniEnt(AbstractOpenSetTTAModel):
    """
    UniEnt
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        template: Optional[List[str]] = None,
        K: Optional[int] = 3,
        clipartt_temp: Optional[float] = 0.01,
        use_cliptta_loss: bool = False,
        use_clipartt_loss: bool = False,
        use_memory: Optional[bool] = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_cliptta_loss = use_cliptta_loss
        self.use_clipartt_loss = use_clipartt_loss
        self.class_names = class_names
        self.template = template
        self.K = K
        self.clipartt_temp = clipartt_temp
        self.use_memory = use_memory

        self.memory = None
        if use_memory:
            self.memory = CCM(num_shots=self.num_shots, num_classes=len(self.class_names), sample_size=self.sample_size)

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        device = images[0].device
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)

        # cos_sim = F.cosine_similarity(image_features[0].unsqueeze(1), self.class_prototypes, dim=2)
        max_cos_sim, _ = (logits[0] * self.logit_scale).softmax(dim=1).max(1)
        min_value = max_cos_sim.min()
        max_value = max_cos_sim.max()

        scores_norm = (max_cos_sim - min_value) / (max_value - min_value)
        ood_scores = 1 - scores_norm.detach().cpu().numpy().reshape(-1, 1)

        gm = GaussianMixture(n_components=2).fit(ood_scores)
        weight = gm.predict_proba(ood_scores)

        # TTA Loss (entropy minimization on ID samples)
        weight = weight if gm.means_[0, 0] < gm.means_[1, 0] else 1 - weight
        weight_id = torch.from_numpy(weight[:, 0]).to(device)
        weight_ood = torch.from_numpy(weight[:, 1]).to(device)

        if not self.use_cliptta_loss or self.use_ood_loss:
            entropys = lib.softmax_entropy(self.logit_scale * logits[0])

        if self.use_cliptta_loss:
            _, pred = logits[0].topk(1, 1, True, True)
            pred_text_features = self.class_prototypes[pred[:, 0]]

            # Obtain new logits (image v.s. new prompt, i.e. size is B x B
            logits_per_image = self.logit_scale * image_features[0] @ pred_text_features.t()
            loss_tta = (weight_id * lib.softmax_entropy(logits_per_image)).mean(0)
        elif self.use_clipartt_loss:
            _, pred = logits[0].topk(self.K, 1, True, True)
            if self.K == 1:
                text_features = self.class_prototypes[pred[:, 0]]
            else:
                text_prompts = lib.getprompt(self.K, pred.cpu().numpy(), self.class_names, self.template[0])
                pred_inputs = clip_tokenize(text_prompts).to(device)

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
            entropys_ind = entropys.mul(weight_id)
            loss_tta = entropys_ind.mean(0)

        # Regularization loss
        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])

        # OOD Loss (entropy maximization on OOD samples)
        if self.use_ood_loss:
            entropys_ood = entropys.mul(weight_ood)
            loss_ood = entropys_ood.mean(0)
        else:
            loss_ood = 0.0

        # Using memory
        if self.use_memory:
            # Updating Memory
            if step == 0:
                with torch.no_grad():
                    logits = self.get_logits(image_features)
                    _, pred = logits[0].topk(1, 1, True, True)
                    scores = self.get_scores(logits, image_features, score_type=self.id_score_type)
                self.memory.update(images[0].cpu().detach(), pred[:, 0].cpu().detach(), scores.cpu().detach())

            # Sampling from memory
            images_mem, _, _ = self.memory.sample()
            image_features = self.get_features([images_mem.to(image_features[0].device)])
            logits = self.get_logits(image_features)
            _, pred = logits[0].topk(1, 1, True, True)
            pred_text_features = self.class_prototypes[pred[:, 0]]

            # Obtain new logits (image v.s. new prompt, i.e. size is B x B
            logits_per_image = self.logit_scale * image_features[0] @ pred_text_features.t()
            loss_tta += (lib.softmax_entropy(logits_per_image)).mean(0)
            loss_reg += lib.softmax_mean_entropy(self.logit_scale * logits[0])

        # Final Loss
        loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg - self.beta_ood * loss_ood
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
