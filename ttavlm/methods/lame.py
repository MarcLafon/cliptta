from typing import Dict, Any, Tuple, List
from typing_extensions import TypeAlias

import torch
from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel

Kwargs: TypeAlias = Dict[str, Any]


class Lame(AbstractOpenSetTTAModel):
    """Tent adapts a model by laplacian optimization during testing.

    Once lamed, a model adapts its logits by updating once.
    """

    def __init__(
        self,
        affinity: str = "knn",
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.affinity = affinity

    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        """Laplacian optimization using image features similarity to refine logits"""
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        probas = torch.softmax(logits[0], dim=1)
        unary = -torch.log(probas + 1e-10)
        if self.affinity == 'knn':
            kernel = kNN_affinity(knn=5)(image_features[0])
        elif self.affinity == 'rbf':
            kernel = rbf_affinity(knn=5)(image_features[0])
        else:
            kernel = linear_affinity()(image_features[0])

        logits = self.laplacian_optimization(unary.type(torch.float32), kernel.type(torch.float32))
        scores = self.get_scores(logits)

        return [logits], scores

    def laplacian_optimization(
            self,
            unary: Tensor,
            kernel: Tensor,
            bound_lambda: int = 1,
            max_steps: int = 100
    ) -> Tensor:
        E_list = []
        oldE = float('inf')
        Y = (-unary).softmax(-1)  # [N, K]
        for i in range(max_steps):
            pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
            exponent = -unary + pairwise
            Y = exponent.softmax(-1)
            E = self.entropy_energy(Y, unary, pairwise, bound_lambda).item()
            E_list.append(E)

            if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
                # logger.info(f'Converged in {i} iterations')
                break
            else:
                oldE = E

        return Y

    def entropy_energy(
            self,
            Y: Tensor,
            unary: Tensor,
            pairwise: Tensor,
            bound_lambda: int,
    ) -> Tensor:
        E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()

        return E


class AffinityMatrix:
    def __init__(self, **kwargs: Kwargs) -> None:
        pass

    def __call__(X, **kwargs: Kwargs) -> Tensor:
        raise NotImplementedError

    def is_psd(self, mat: Tensor) -> Tuple[Tensor, float]:
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]

        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat: Tensor) -> Tensor:
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int) -> None:
        self.knn = knn

    def __call__(self, X: Tensor) -> Tensor:
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, **kwargs: Kwargs) -> None:
        self.k = kwargs['knn']

    def __call__(self, X: Tensor) -> Tensor:
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))

        return rbf


class linear_affinity(AffinityMatrix):
    def __call__(self, X: Tensor) -> Tensor:
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())
