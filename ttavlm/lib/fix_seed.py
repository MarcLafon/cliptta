import random
import numpy as np
import torch

from ttavlm.lib.logger import LOGGER


def fix_seed(seed: int, backend: bool = True) -> None:
    LOGGER.info(f"Setting the seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
