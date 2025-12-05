from typing import Tuple

import numpy as np

from sklearn import metrics
from scipy import interpolate


def get_ood_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    auroc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return auroc, float(interpolate.interp1d(tpr, fpr)(0.95))
