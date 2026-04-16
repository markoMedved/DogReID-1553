import random
from torch.utils.data.sampler import Sampler
from collections import defaultdict


class RandomIdentitySampler(Sampler):

    def __init__(self, dataset, num_ids=4, num_instances=2):
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_instances = num_instances

        self.index_dic = defaultdict(list)

        for idx in range(len(dataset)):
            label = dataset.get_label(idx)
            self.index_dic[label].append(idx)

        self.pids = list(self.index_dic.keys())

        # how many samples per epoch
        self.length = 0
        for pid in self.pids:
            self.length += len(self.index_dic[pid])

    def __iter__(self):
        batch_indices = []

        random.shuffle(self.pids)

        for i in range(0, len(self.pids), self.num_ids):

            selected_pids = self.pids[i:i+self.num_ids]

            for pid in selected_pids:
                idxs = self.index_dic[pid]

                if len(idxs) >= self.num_instances:
                    chosen = random.sample(idxs, self.num_instances)
                else:
                    chosen = random.choices(idxs, k=self.num_instances)

                batch_indices.extend(chosen)

        return iter(batch_indices)

    def __len__(self):
        return self.length