from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
import numpy as np


def find_sequence_length(x):
    """Find the sequence length."""
    nnz = np.sign(np.max(np.abs(x), axis=2))
    nnz_lens = np.sum(nnz.astype(np.int32), axis=1)
    return nnz_lens.astype(np.int32)


class Data(object):
    """Creat data class."""

    def __init__(self, is_train, datapath, batch_size):
        """Initialize dataset."""
        self.is_train = is_train
        dataset = np.load(datapath)
        ids = dataset["ids"]
        self.ids = ids[ids < 30]
        self.data = dataset["data"][ids < 30]
        self.num_examples = len(self.ids)  # dataset["num_examples"]
        self.num_classes = 30  # dataset["num_classes"]

    def get_next(self, indices, max_same=1, max_diff=1):
        """For training."""
        batch_size = len(indices)

        anchor_indices = indices
        same_partition = np.arange(batch_size, dtype=np.int32)
        diff_partition = np.array([], dtype=np.int32)

        cur = batch_size
        for idx in anchor_indices:  # -- get same indices
            same_indices = np.where(self.ids == self.ids[idx])[0]
            same_indices = np.random.permutation(same_indices[same_indices != idx])[:max_same]
            indices = np.concatenate((indices, same_indices))
            same_partition = np.concatenate((same_partition, np.full_like(same_indices, cur)))
            cur += 1

        for ii, idx in enumerate(anchor_indices):  # -- get diff indices
            diff_indices = np.where(self.ids != self.ids[idx])[0]
            diff_indices = np.random.permutation(diff_indices)[:max_diff]
            indices = np.concatenate((indices, diff_indices))
            same_partition = np.concatenate((same_partition, np.arange(cur, cur + len(diff_indices))))
            diff_partition = np.concatenate((diff_partition, np.full_like(diff_indices, ii)))
            cur += len(diff_indices)

        x = self.data[indices]
        ts = find_sequence_length(x)

        return x, ts, same_partition, diff_partition
