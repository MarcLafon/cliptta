from typing import Dict, Any, List, Tuple
from typing_extensions import TypeAlias

from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel

Kwargs: TypeAlias = Dict[str, Any]


class SourceModel(AbstractOpenSetTTAModel):
    """ """

    def __init__(
        self,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        scores = self.get_scores(logits, image_features)
        return logits, scores
