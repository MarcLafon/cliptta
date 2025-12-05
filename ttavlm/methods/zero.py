from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel

Kwargs: TypeAlias = Dict[str, Any]


class Zero(AbstractOpenSetTTAModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        zero_gamma: float = 0.3,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.zero_gamma = zero_gamma

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
            image_features = self.get_features(images)
            logits = self.get_logits(image_features)
            scores = self.get_scores(logits)
            logits = torch.stack(logits, dim=0)  # N_views, Batch_size, N_classes

        # Confidence filter
        l_filt, _, sorted_idx = self.confidence_filter(logits, top=self.zero_gamma, return_idx=True)
        # Zero-out the temperature, marginalize and predict
        zero_temp = torch.finfo(l_filt.dtype).eps
        p_bar = (l_filt / zero_temp).softmax(-1).sum(0)

        # Check if we have to break ties in some way
        for i in range(logits.shape[1]):
            max_counts, scalar_pred = p_bar[i, :].max(-1)
            ties = [scalar_pred]
            for j in range(p_bar.shape[-1]):
                if j == scalar_pred:
                    continue
                if p_bar[i, j] == max_counts:
                    ties.append(j)

            # if so, break ties greedily
            if len(ties) > 1:
                k = int(len(images) * self.zero_gamma)
                sorted_l = logits[:, i][sorted_idx[:, i]]
                scalar_pred = self.greedy_break(ties, sorted_l[k:])
                p_bar[i, scalar_pred] += 1

        # Get final logits and OOD scores
        with torch.no_grad():
            image_features = self.get_features(images)
            logits = self.get_logits(image_features)[0]
            scores = self.get_scores([logits])

        return logits, scores, p_bar

    def softmax_entropy(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -((x + eps).softmax(2) * (x + eps).log_softmax(2)).sum(2)

    def confidence_filter(self, logits: Tensor, top: float, return_idx: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        batch_entropy = self.softmax_entropy(logits * self.logit_scale)  # N_views, Batch_size, 1
        full_idx = torch.argsort(batch_entropy, dim=0, descending=False)
        m = max(1, int(batch_entropy.size()[0] * top))
        filt_idx = full_idx[:m]
        filt_logits = torch.stack([logits[:, i][filt_idx[:, i]] for i in range(filt_idx.shape[1])], dim=1)
        if not return_idx:
            return filt_logits
        return filt_logits, filt_idx, full_idx

    def greedy_break(self, ties: List, logits: torch.Tensor) -> Tensor:
        ties_tensor = torch.tensor(ties, dtype=torch.int, device=logits.device)
        preds = torch.argmax(logits, dim=1)
        for pred in preds:
            if pred in ties_tensor:
                return pred
        return self.break_sample_tie(ties, logit=logits[0])

    def break_sample_tie(self, ties: List, logit: torch.Tensor) -> Tensor:
        ties = torch.tensor(ties, dtype=torch.int64, device=logit.device)
        logit[~ties] = -torch.inf
        scalar_pred = torch.argmax(logit, dim=-1)
        return scalar_pred
