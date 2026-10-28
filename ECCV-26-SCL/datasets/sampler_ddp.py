import copy
import math
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


def shared_random_seed():
    seed = torch.randint(0, 2**31 - 1, (1,), device="cuda" if torch.cuda.is_available() else "cpu")
    if dist.get_rank() != 0:
        seed.zero_()
    dist.broadcast(seed, src=0)
    return int(seed.item())


class RandomIdentitySampler_DDP(Sampler):
    """Distributed identity sampler for triplet training."""

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, sample in enumerate(self.data_source):
            _, pid, *_ = sample
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        self.length //= self.world_size

    def __iter__(self):
        np.random.seed(shared_random_seed())
        final_idxs = self._sample_list()
        length = int(math.ceil(len(final_idxs) / self.world_size))
        final_idxs = self._fetch_rank_indices(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def _fetch_rank_indices(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = length // self.mini_batch_size
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            start = self.mini_batch_size * self.rank + self.mini_batch_size * i
            end = min(start + self.mini_batch_size, total_num)
            index_target.extend(range(start, end))
        if not index_target:
            return list(np.array(final_idxs)[self.rank:total_num:self.world_size])
        return list(np.array(final_idxs)[np.array(index_target, dtype=np.int64)])

    def _sample_list(self):
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}
        batch_indices = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(
                avai_pids,
                self.num_pids_per_batch,
                replace=False,
            ).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(
                            idxs,
                            size=self.num_instances,
                            replace=True,
                        ).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length
