# flake8: noqa

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class CLIPFeaturesDataset(Dataset):
    def __init__(self, features_path, labels_path, feature_dim):
        features_size = os.path.getsize(features_path)

        self.num_samples = features_size // (
            feature_dim * 4
        )  # 4 bytes per float32 value
        self.feature_dim = feature_dim

        self.features = np.memmap(
            features_path,
            dtype="float32",
            mode="r",
            shape=(self.num_samples, self.feature_dim),
        )
        self.labels = np.memmap(
            labels_path, dtype="int64", mode="r", shape=(self.num_samples,)
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
