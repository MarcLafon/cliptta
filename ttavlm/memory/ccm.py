from typing import Tuple, List

import torch

import ttavlm.lib as lib


class CCM:
    """
    Class-wise Confident Memory
    """

    def __init__(
        self,
        num_shots: int = 4,
        num_classes: int = 10,
        sample_size: int = 256,
        img_size: int = 224,
        n_channels: int = 3,
    ) -> None:
        self.num_shots = num_shots
        self.num_classes = num_classes
        self.sample_size = sample_size
        self.img_size = img_size
        self.n_channels = n_channels
        lib.LOGGER.info(f"Instanciating CCM memory of size {num_shots * num_classes}")
        if sample_size > num_shots * num_classes:
            lib.LOGGER.info(f"Warning: sample_size={sample_size} is greater than num_shots * num_classes = {num_shots * num_classes}, sample_size is now {num_shots * num_classes}.")
            self.sample_size = int(num_shots * num_classes)

        if sample_size < num_classes:
            lib.LOGGER.info(f"Warning: sample_size={sample_size} is lower than num_classes, some classes will not be represented in the sampled batch.")

        self.data = [torch.empty((0, self.n_channels, self.img_size, self.img_size)) for _ in range(self.num_classes)]
        self.labels = [torch.empty((0,)) for _ in range(self.num_classes)]
        self.confidences = [torch.empty((0,)) for _ in range(self.num_classes)]

    def __len__(self) -> int:
        return sum([len(data_class) for data_class in self.data])

    @property
    def current_size(self) -> int:
        return self.__len__()

    @property
    def per_class_current_size(self) -> List[int]:
        return [len(data_class) for data_class in self.data]

    def update(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        confidences: torch.Tensor,
    ) -> None:
        for i in range(self.num_classes):
            indices_i = labels == i
            if indices_i.sum().item() == 0:
                continue

            self.data[i] = torch.cat([self.data[i], images[indices_i]])
            self.labels[i] = torch.cat([self.labels[i], labels[indices_i]])
            self.confidences[i] = torch.cat([self.confidences[i], confidences[indices_i]])

            if self.per_class_current_size[i] > self.num_shots:
                _, idx_topk = self.confidences[i].topk(k=self.num_shots)
                self.data[i] = self.data[i][idx_topk]
                self.confidences[i] = self.confidences[i][idx_topk]
                self.labels[i] = self.labels[i][idx_topk]

    def reset(self) -> None:
        self.data = [torch.empty((0, self.n_channels, self.img_size, self.img_size)) for _ in range(self.num_classes)]
        self.labels = [torch.empty((0,)) for _ in range(self.num_classes)]
        self.confidences = [torch.empty((0,)) for _ in range(self.num_classes)]

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_class_sample_size = max(1, int(self.sample_size / self.num_classes))
        images, labels, scores = [], [], []
        for i in range(self.num_classes):
            if self.per_class_current_size[i] > per_class_sample_size:
                random_choice = torch.randperm(self.per_class_current_size[i], dtype=torch.int32, device="cpu")[:per_class_sample_size].long()
                images.append(self.data[i][random_choice])
                labels.append(self.labels[i][random_choice])
                scores.append(self.confidences[i][random_choice])
            else:
                images.append(self.data[i])
                labels.append(self.labels[i])
                scores.append(self.confidences[i])

        images = torch.cat(images)
        labels = torch.cat(labels)
        scores = torch.cat(scores)

        return images, labels, scores
