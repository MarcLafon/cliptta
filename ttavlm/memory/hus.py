from typing import Dict, List
from collections import OrderedDict

import random
import numpy as np

from torch import Tensor


class HUS:
    def __init__(self,
                 capacity: int = 1,
                 threshold: float = None,
                 num_classes: int = 10,
                 ) -> None:
        self.data = [[[], [], []] for _ in range(num_classes)]
        self.counter = [0] * num_classes
        self.marker = [''] * num_classes
        self.capacity = capacity
        self.threshold = threshold   # * math.log(num_classes)
        self.num_classes = num_classes

    def set_memory(self, state_dict: OrderedDict) -> None:
        self.data = [[logit[:] for logit in logits] for logits in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self) -> Dict:
        dic = {}
        dic['data'] = [[logit[:] for logit in logits] for logits in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold

        return dic

    def print_class_dist(self) -> None:
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self) -> None:
        occupancy_per_class = [0] * self.num_classes
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self) -> List:
        data = self.data

        tmp_data = [[], []]
        for data_per_cls in data:
            feats, cls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
        return tmp_data

    def get_occupancy(self) -> int:
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self) -> List:
        occupancy_per_class = [0] * self.num_classes
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance: List) -> None:
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[2] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self) -> List:
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self) -> int:
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data: List) -> int:
        return random.randrange(0, len(data))

    def remove_instance(self, cls: int) -> bool:
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][2])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][2])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats: Tensor,
                    cls: List,
                    aux: List,
                    ) -> None:
        self.data = [[[], [], []] for _ in range(self.num_classes)]  # feat, pseudo_cls, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][3].append(aux[i])

    def reset(self) -> None:
        self.data = [[[], [], []] for _ in range(self.num_classes)]
