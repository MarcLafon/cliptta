import torch
import torch.nn.functional as F


def softmax_entropy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -((x + eps).softmax(1) * (x + eps).log_softmax(1)).sum(1)


def softmax_mean_entropy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = (x + eps).softmax(1).mean(0)
    return -(x * torch.log(x + eps)).sum()


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Entropy of probability distribution."""
    return -(p * torch.log(p)).sum(1)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Entropy of probability distribution."""
    return (-targets * F.log_softmax(logits, dim=-1)).sum(1)
