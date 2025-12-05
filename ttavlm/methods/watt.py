from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy
from sklearn.mixture import GaussianMixture

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.models.clip import clip
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class Watt(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        class_names: List[str],
        template: List[str],
        avg_type: str,
        meta_reps: int,
        reps: int,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_names = class_names
        self.template = template
        self.avg_type = avg_type
        self.reps = reps
        self.meta_reps = meta_reps
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

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
        if not hasattr(self, "text_features"):
            self.text_features = self.extract_text_embeddings(device=images[0].device, average=False)

        model_state = avg_state_dict = self.copy_state(self.model)

        for m in range(self.meta_reps):
            all_weights = []
            if self.avg_type == "sequential":
                if m == 0:
                    self.model.load_state_dict(model_state, strict=False)
                else:
                    self.model.load_state_dict(avg_state_dict, strict=False)

            for class_prototypes in self.text_features:
                if self.avg_type == "parallel":
                    if m == 0:
                        self.model.load_state_dict(model_state, strict=False)
                    else:
                        self.model.load_state_dict(avg_state_dict, strict=False)

                # Adapting for the current template
                for _ in range(self.reps):
                    # Getting prediction-based text features
                    with torch.no_grad():
                        image_features = self.get_features(images)
                    logits = self.get_logits(image_features, class_prototypes)
                    probs = logits[0].softmax(1)
                    _, pred = probs.topk(1, 1, True, True)
                    pred_inputs = class_prototypes[pred[:, 0]]

                    # Adapting the model
                    image_features = self.get_features(images)
                    logits = self.logit_scale * image_features[0] @ pred_inputs.t()
                    images_similarity = image_features[0] @ image_features[0].t()
                    texts_similarity = pred_inputs @ pred_inputs.t()
                    targets = F.softmax(((images_similarity + texts_similarity) / 2) / 0.01, dim=-1)
                    loss = self.loss_fn(logits.t(), targets.t())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Collecting current step's weights
                weights = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        for np, p in module.named_parameters():
                            if np in ["weight", "bias"]:
                                weights[f"{name}.{np}"] = deepcopy(p)
                all_weights.append(weights)
            avg_state_dict = self.weight_average(all_weights)
        self.model.load_state_dict(avg_state_dict, strict=False)

        # Get final logits and OOD scores
        with torch.no_grad():
            image_features = self.get_features(images)
            logits = self.get_logits(image_features)
            scores = self.get_scores(logits)

        return logits, scores

    def extract_text_embeddings(self, device: str, average: bool = True) -> Tensor:
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """

        with torch.no_grad():
            text_features = []
            for template in self.template:
                texts = [template.format(class_name) for class_name in self.class_names]
                texts = clip.tokenize(texts).to(device)
                class_embeddings = self.clip_text_encoder.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)

            text_features = torch.stack(text_features, dim=0).to(device)

            if average:
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def weight_average(self, all_weights: Dict) -> Dict:
        """
        Compute the average of the weights from multiple models.

        Args:
            all_weights: List of state dictionaries from different models.

        Returns:
            avg_state_dict: Averaged state dictionary.
        """
        K = len(all_weights)
        avg_state_dict = OrderedDict()
        for param_name, param in all_weights[0].items():
            avg_param = sum(sd[param_name] for sd in all_weights) / K
            avg_state_dict[param_name] = avg_param
        return avg_state_dict


class WattOtsu(Watt):
    def __init__(self, *args: List[Any], **kwargs: Kwargs) -> None:
        super(WattOtsu, self).__init__(*args, **kwargs)

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
        images0 = copy.deepcopy(images[0]).detach().cpu()

        if not hasattr(self, "text_features"):
            self.text_features = self.extract_text_embeddings(device=images[0].device, average=False)

        model_state = avg_state_dict = self.copy_state(self.model)

        for m in range(self.meta_reps):
            all_weights = []
            if self.avg_type == "sequential":
                if m == 0:
                    self.model.load_state_dict(model_state, strict=False)
                else:
                    self.model.load_state_dict(avg_state_dict, strict=False)

            for class_prototypes in self.text_features:
                if self.avg_type == "parallel":
                    if m == 0:
                        self.model.load_state_dict(model_state, strict=False)
                    else:
                        self.model.load_state_dict(avg_state_dict, strict=False)

                # Adapting for the current template
                for _ in range(self.reps):
                    image_features = self.get_features(images)
                    logits = self.get_logits(image_features, class_prototypes)

                    # Regularization loss
                    if self.beta_reg != 0:
                        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])
                    else:
                        loss_reg = 0.0

                    # OOD loss computation
                    scores = self.get_scores(logits, image_features[0])
                    loss_ood = self.get_otsu_loss(scores)

                    # OOD-ID separation
                    images, image_features = self.filter_id(images, image_features, scores, self.alpha if self.update_alpha else None)
                    logits = self.get_logits(image_features, class_prototypes)
                    probs = logits[0].softmax(1)
                    _, pred = probs.topk(1, 1, True, True)
                    pred_inputs = class_prototypes[pred[:, 0]]
                    # pred_inputs = torch.cat([class_prototypes[c,] for c in pred])

                    # Adapting the model
                    image_features = self.get_features(images)
                    logits = self.logit_scale * image_features[0] @ pred_inputs.t()
                    images_similarity = image_features[0] @ image_features[0].t()
                    texts_similarity = pred_inputs @ pred_inputs.t()
                    targets = F.softmax(((images_similarity + texts_similarity) / 2) / 0.01, dim=-1)
                    loss_tta = self.loss_fn(logits.t(), targets.t())

                    loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg + self.beta_ood * loss_ood
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Collecting current step's weights
                weights = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        for np, p in module.named_parameters():
                            if np in ["weight", "bias"]:
                                weights[f"{name}.{np}"] = deepcopy(p)
                all_weights.append(weights)
            avg_state_dict = self.weight_average(all_weights)
        self.model.load_state_dict(avg_state_dict, strict=False)

        # Get final logits and OOD scores
        with torch.no_grad():
            image_features = self.get_features([images0.to(device)])
            logits = self.get_logits(image_features)
            scores = self.get_scores(logits)

        return logits, scores


class WattUniEnt(Watt):
    def __init__(self, *args: List[Any], **kwargs: Kwargs) -> None:
        super(WattUniEnt, self).__init__(*args, **kwargs)

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
        if not hasattr(self, "text_features"):
            self.text_features = self.extract_text_embeddings(device=images[0].device, average=False)

        model_state = avg_state_dict = self.copy_state(self.model)

        for m in range(self.meta_reps):
            all_weights = []
            if self.avg_type == "sequential":
                if m == 0:
                    self.model.load_state_dict(model_state, strict=False)
                else:
                    self.model.load_state_dict(avg_state_dict, strict=False)

            for class_prototypes in self.text_features:
                if self.avg_type == "parallel":
                    if m == 0:
                        self.model.load_state_dict(model_state, strict=False)
                    else:
                        self.model.load_state_dict(avg_state_dict, strict=False)

                # Adapting for the current template
                for _ in range(self.reps):
                    # Getting prediction-based text features
                    image_features = self.get_features(images)
                    logits = self.get_logits(image_features, class_prototypes)

                    # Regularization loss
                    if self.beta_reg != 0:
                        loss_reg = lib.softmax_mean_entropy(self.logit_scale * logits[0])
                    else:
                        loss_reg = 0.0

                    # OOD loss computation
                    scores = self.get_scores(logits, image_features[0]).detach().cpu().numpy().reshape(-1, 1)
                    gm = GaussianMixture(n_components=2).fit(scores)
                    weight = gm.predict_proba(scores)

                    weight = weight if gm.means_[0, 0] < gm.means_[1, 0] else 1 - weight
                    weight_ood = torch.from_numpy(weight[:, 1]).to(logits[0].device)
                    entropys = lib.softmax_entropy(self.logit_scale * logits[0])
                    entropys_ood = entropys.mul(weight_ood)
                    loss_ood = entropys_ood.mean(0)

                    probs = logits[0].softmax(1)
                    _, pred = probs.topk(1, 1, True, True)
                    pred_inputs = class_prototypes[pred[:, 0]]
                    logits = self.logit_scale * image_features[0] @ pred_inputs.t()
                    images_similarity = image_features[0] @ image_features[0].t()
                    texts_similarity = pred_inputs @ pred_inputs.t()
                    targets = F.softmax(((images_similarity + texts_similarity) / 2) / 0.01, dim=-1)
                    loss_tta = self.loss_fn(logits.t(), targets.t())

                    loss = self.beta_tta * loss_tta - self.beta_reg * loss_reg - self.beta_ood * loss_ood
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Collecting current step's weights
                weights = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        for np, p in module.named_parameters():
                            if np in ["weight", "bias"]:
                                weights[f"{name}.{np}"] = deepcopy(p)
                all_weights.append(weights)
            avg_state_dict = self.weight_average(all_weights)
        self.model.load_state_dict(avg_state_dict, strict=False)

        # Get final logits and OOD scores
        with torch.no_grad():
            image_features = self.get_features(images)
            logits = self.get_logits(image_features)
            scores = self.get_scores(logits)

        return logits, scores
