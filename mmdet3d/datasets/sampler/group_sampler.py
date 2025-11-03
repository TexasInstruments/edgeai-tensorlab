# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import copy
import itertools
import numpy as np
import torch
from typing import Iterator, Optional, Sized
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class GroupEachSampleInBatchSampler(Sampler):
    """ Implmentation based on InfiniteGroupEachSampleInBatchSampler for EpochBasedRunner

        Basically, we want every sample to be from its own group.
        If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
        its own group. Shuffling is only done for group order, not done within groups
    """

    def __init__(self,
                 dataset,
                 shuffle=True,
                 skip_prob: float = 0.0,
                 sequence_flip_prob: float = 0.0,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        assert hasattr(dataset, 'flag')

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        self.batch_size = dataset.batch_size
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = self.batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}

        # Minium data size for each GPU is
        #  (Total number of groups per batch) * batch_size * (smallest group size)
        # Number of groups per GPU
        self.device_groups_num = self.groups_num // self.world_size
        # Number of groups per batch in a GPU
        self.batch_groups_num = math.ceil(self.device_groups_num / self.batch_size)
        # Average number of samples in a group
        self.avg_group_samples = sum(self.group_sizes) // self.groups_num
        self.num_samples_batch = self.batch_groups_num * self.avg_group_samples
        # Total data size
        self.size = self.batch_size * self.num_samples_batch

        # For Sparse4D
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob

    # generate interation indices for every batch 
    def generate_iteration_indices(self):
        self.infinite_group_indices_list = self._infinite_group_indices()

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx, self.infinite_group_indices_list)
            for local_sample_idx in range(self.batch_size)]

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        buffer_per_local_sample = [[] for _ in range(self.batch_size)]

        indices = []
        samples_count = [0] * self.batch_size
        skip_count = [0] * self.batch_size
        while True:
            # Check if skip flag improves training
            skip = (np.random.uniform() < self.skip_prob)
            for batch_idx in range(self.batch_size):
                if len(buffer_per_local_sample[batch_idx]) == 0:
                    try:
                        # Finished current group, refill with next group
                        new_group_idx = next(group_indices_per_global_sample_idx[batch_idx])
                    except StopIteration:
                        #Recreate the iterator to reset it
                        group_indices_per_global_sample_idx[batch_idx] = \
                            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + batch_idx, self.infinite_group_indices_list)
                        new_group_idx = next(group_indices_per_global_sample_idx[batch_idx])

                    # To load only self.num_samples_batch samples
                    samples_to_copy = min(len(self.group_idx_to_sample_idxs[new_group_idx]), self.num_samples_batch - samples_count[batch_idx])
                    buffer_per_local_sample[batch_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx][:samples_to_copy])

                    if np.random.uniform() < self.sequence_flip_prob:
                        buffer_per_local_sample[batch_idx] = \
                            buffer_per_local_sample[batch_idx][::-1]

                if skip and len(buffer_per_local_sample[batch_idx]) > 1:
                    buffer_per_local_sample[batch_idx].pop(0)
                    skip_count[batch_idx] += 1

                indices.append(buffer_per_local_sample[batch_idx].pop(0))
                samples_count[batch_idx] += 1

            # Once enough # of data is loaded for evey batch, stop
            # With skip_prob != 0.0, replace all with any
            if self.skip_prob > 0.0:
                if any([x == self.num_samples_batch - y for x, y in zip(samples_count, skip_count)]):
                    break
            else:
                if all([x == self.num_samples_batch for x in samples_count]):
                    break

        if self.skip_prob == 0.0:
            # The following does not hold with skip_prob != 0.0
            assert len(indices) == self.size
        return indices

    # Shuffle group indices
    def _infinite_group_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return torch.randperm(self.groups_num, generator=g).tolist()
        else:
            return torch.arange(self.groups_num).tolist()


    def _group_indices_per_global_sample_idx(self, global_sample_idx, infinite_group_indices_list):
        return itertools.islice(infinite_group_indices_list,
                                global_sample_idx,
                                None,
                                self.global_batch_size)

    def __iter__(self) -> Iterator[int]:
        self.indices = self.generate_iteration_indices()
        return iter(self.indices)

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

