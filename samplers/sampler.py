import random
from torch.utils.data.sampler import Sampler
from collections import defaultdict


class RandomIdentitySampler(Sampler):

    def __init__(self, dataset, num_ids=4, num_instances=2):

        self.dataset = dataset
        self.num_ids = num_ids
        self.num_instances = num_instances

        self.index_dic = defaultdict(list)

        # ❌ NO dataset[idx]
        # ✔ only metadata
        for idx in range(len(dataset)):
            label = dataset.get_label(idx)
            self.index_dic[label].append(idx)

        self.pids = list(self.index_dic.keys())

    def __iter__(self):

        batch_indices = []

        random.shuffle(self.pids)

        for pid in self.pids:

            idxs = self.index_dic[pid]

            if len(idxs) >= self.num_instances:
                idxs = random.sample(idxs, self.num_instances)
            else:
                idxs = random.choices(idxs, k=self.num_instances)

            batch_indices.extend(idxs)

        return iter(batch_indices)

    def __len__(self):
        return len(self.dataset)