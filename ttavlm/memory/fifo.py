# flake8: noqa
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch


class FIFO_sotta:
    def __init__(self,
                 capacity: int,
                 ) -> None:
        self.data = [[], []]
        self.capacity = capacity

    def set_memory(self,
                   state_dict: OrderedDict,
                   ) -> None:
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self) -> Dict:
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity

        return dic

    def get_memory(self) -> List:
        return self.data

    def get_occupancy(self) -> int:
        return len(self.data[0])

    def add_instance(self, instance: List) -> None:
        assert (len(instance) == 2)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self) -> None:
        for dim in self.data:
            dim.pop(0)
        pass


class FIFO:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.data = [torch.empty((0, self.n_channels, self.img_size, self.img_size)) for _ in range(self.num_classes)]
        self.labels = [torch.empty((0,)) for _ in range(self.num_classes)]
        self.confidences = [torch.empty((0,)) for _ in range(self.num_classes)]

    def update(self, item):
        if self.data is None:
            self.data = item
        else:
            self.data = torch.cat([self.data, item], dim=0)
        if len(self.data) > self.max_len:
            self.data = self.data[-self.max_len:]

    def __len__(self):
        return len(self.data)

    def sample(self) -> Tuple[torch.Tensor]:
        return self.data

    def reset(self):
        self.data = [torch.empty((0, self.n_channels, self.img_size, self.img_size)) for _ in range(self.num_classes)]
        self.labels = [torch.empty((0,)) for _ in range(self.num_classes)]
        self.confidences = [torch.empty((0,)) for _ in range(self.num_classes)]
