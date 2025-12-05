from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cross_entropy
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.transforms import get_tta_transforms
from ttavlm.memory import CSTU
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class RoTTA(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        capacity: int,
        update_frequency: int,
        lambda_u: List[float],
        lambda_t: List[float],
        alpha_rotta: float,
        nu: float,
        use_tta: bool,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mem = CSTU(capacity, len(self.class_prototypes), lambda_t=lambda_t, lambda_u=lambda_u)
        self.update_frequency = update_frequency
        self.transforms = get_tta_transforms(n_pixels=224)
        self.alpha_rotta = alpha_rotta
        self.nu = nu
        self.use_tta = use_tta
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach()
        self.current_instance = 0

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
        with torch.no_grad():
            image_features_ema = self.get_features(images, self.model_ema)
            logits_ema = self.get_logits(image_features_ema)
            prediction_ema = logits_ema[0].softmax(1)
            pseudo_label = torch.argmax(prediction_ema, dim=1)
            entropy = lib.softmax_entropy(logits_ema[0])

        # Updating memory
        self.update_memory(images, pseudo_label, entropy)

        if self.current_instance % self.update_frequency == 0:
            self.update_model(self.model)
            self.current_instance = 0

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_logits(image_features)
                scores = self.get_scores(logits)
        else:
            logits, scores = None, None

        return logits, scores

    def update_memory(self, images: List[Tensor], pseudo_label: Tensor, entropy: Tensor) -> None:
        for i, data in enumerate(images[0]):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

    def update_model(self, model: nn.Module) -> None:
        model.train()
        self.model_ema.train()
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            if self.use_tta:
                sup_data = self.transforms(sup_data)
            strong_sup_features = self.get_features([sup_data], self.model_ema)
            ema_sup_logits = self.get_logits(strong_sup_features)
            model_sup_features = self.get_features([sup_data])
            model_sup_logits = self.get_logits(model_sup_features)
            instance_weight = self.timeliness_reweighting(ages)
            l_sup = (instance_weight * cross_entropy(model_sup_logits[0], ema_sup_logits[0], reduction='none')).mean()

        loss = l_sup
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.ema_model = self.update_ema_variables(self.model_ema, self.model, self.nu)

    def update_ema_variables(self,
                             ema_model: nn.Module,
                             model: nn.Module,
                             nu: float
                             ) -> nn.Module:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def timeliness_reweighting(self, ages: List) -> float:
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))

    def get_named_submodule(self, model: nn.Module, sub_name: str) -> nn.Module:
        names = sub_name.split(".")
        module = model
        for name in names:
            module = getattr(module, name)

        return module

    def set_named_submodule(self, model: nn.Module, sub_name: str, value: nn.Module) -> None:
        names = sub_name.split(".")
        module = model
        for i in range(len(names)):
            if i != len(names) - 1:
                module = getattr(module, names[i])
            else:
                setattr(module, names[i], value)
