import numpy as np
import torch
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pandas import DataFrame


class Transform(ABC):
    @abstractmethod
    def __call__(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        pass


class DenseIndexing(Transform):
    def __init__(self, item_map_df: DataFrame):
        self.item_map: Dict[Any, int] = self.refine_map_from(item_map_df)
        self.indexer = np.vectorize(self.item_map.get)

    @staticmethod
    def refine_map_from(item_map: DataFrame) -> Dict[int, int]:
        return {Id: idx for Id, idx in item_map[["item_id", "item_idx"]].values}

    def __call__(self, samples):
        inputs, targets = samples["inputs"], samples["targets"]

        samples.update({
            "inputs": self.indexer(inputs),
            "targets": self.indexer(targets)
        })
        return samples


class Masking(Transform):
    def __init__(self):
        self.batch_size = None

    def get_mask(self, indices):
        mask = np.zeros(shape=(self.batch_size, 1))
        mask[indices, :] = 1.0
        return mask

    def __call__(self, samples):
        self.batch_size = len(samples["inputs"])

        session_change_idx = samples["session_change"]
        user_change_idx = samples['user_change']

        samples.update({
            "session_change": self.get_mask(session_change_idx),
            "user_change": self.get_mask(user_change_idx),
        })
        return samples


class ToTensor(Transform):
    def __init__(self, device):
        self.device = device

    def __call__(self, samples):
        return {
            "inputs": torch.LongTensor(samples["inputs"]).to(self.device),
            "targets": torch.LongTensor(samples["targets"]).to(self.device),
            "session_change": torch.FloatTensor(samples["session_change"]).to(self.device),
            "user_change": torch.FloatTensor(samples["user_change"]).to(self.device),
        }


class Sampler(object):
    def __init__(self, df, n_samples, sample_alpha=0.75, sample_store=10000):
        self.df = df
        self.n_samples = n_samples
        self.sample_alpha = sample_alpha  # 3/4
        self.sample_store = sample_store
        self.generate_size = sample_store // n_samples

        self.neg_samples = self._generate_neg_samples(self.generate_size)
        self.sample_pointer = -1

    def __next__(self):
        self.sample_pointer += 1

        if self.sample_pointer == self.generate_size:
            self.neg_samples = self._generate_neg_samples(self.generate_size)
            self.sample_pointer = 0

        return self.neg_samples[self.sample_pointer]

    def __iter__(self):
        return self

    def _init_sample(self):
        self.neg_samples = self._generate_neg_samples(self.generate_size)
        self.sample_pointer = 0

    def _generate_neg_samples(self, length):
        n_item = self._pop.size
        sample_size = length * self.n_samples

        if self.sample_alpha > 0:
            draw = np.random.rand(sample_size)
            sample = np.searchsorted(self._pop, draw)
        else:
            sample = np.random.choice(n_item, size=sample_size)

        if length > 1:
            sample = sample.reshape((length, self.n_samples))
        return sample

    @property
    def _pop(self):
        """ Calculate the distribution P(w_i) of negative sampling """
        prob = self.df["item_id"].value_counts() ** self.sample_alpha
        return prob.cumsum() / prob.sum()


class DataLoader(IterableDataset):
    def __init__(self, args, df, transforms: Optional[List[Transform]] = None):
        super(DataLoader).__init__()
        self.df = df.sort_values(by=["user_id", "timestamp"])
        self.batch_size = args.batch_size
        self.neg_sampler = Sampler(df, args.n_samples)
        self.transforms = transforms

        self.users = df["user_id"].unique()
        self.sessions = df["session_id"].unique()

        self.session_offset = np.r_[0, self.df.groupby("session_id", sort=False).size().cumsum().values]
        self.num_session_for_user = np.r_[0, self.df.groupby("user_id", sort=False)["session_id"].nunique().cumsum().values]
        self.user_offset = self.session_offset[self.num_session_for_user]

        self.user_idx_arr = np.arange(len(self.users))
        self.session_idx_arr = np.arange(len(self.sessions))

    def __iter__(self):
        batch_size = min(self.batch_size, len(self.users))

        user_iter = np.arange(batch_size)
        max_user_iter = np.max(user_iter)

        user_start = self.user_offset[self.user_idx_arr[user_iter]]
        user_end = self.user_offset[self.user_idx_arr[user_iter] + 1]

        session_iter = self.num_session_for_user[user_iter]

        session_start = self.session_offset[session_iter]
        session_end = self.session_offset[session_iter + 1]

        session_change = []
        user_change = []
        finished = False

        while not finished:
            min_session_interval = np.min(session_end - session_start)

            for i in range(min_session_interval-1):
                inputs = self.df["item_id"].values[session_start+i]
                targets = self.df["item_id"].values[session_start+i+1]

                # if self.n_samples > 0:
                #     targets = np.hstack([targets, next(self.neg_sampler)])

                samples = {
                    "inputs": inputs,
                    "targets": targets,
                    "session_change": session_change,
                    "user_change": user_change,
                }

                if isinstance(self.transforms, list):
                    for transform in self.transforms:
                        samples = transform(samples)

                yield samples

            session_start += min_session_interval - 1
            session_change = np.arange(batch_size)[session_end - session_start <= 1]

            for idx in session_change:
                if session_iter[idx] + 1 >= len(self.sessions):
                    finished = True
                    break
                session_iter[idx] += 1

                session_start[idx] = self.session_offset[session_iter[idx]]
                session_end[idx] = self.session_offset[session_iter[idx] + 1]

            user_change_idx = np.arange(batch_size)[user_end - session_start <= 0]

            for idx in user_change_idx:
                if max_user_iter + 1 >= len(self.users):
                    continue
                max_user_iter += 1

                user_iter[idx] = max_user_iter
                user_start[idx] = self.user_offset[max_user_iter]
                user_end[idx] = self.user_offset[max_user_iter + 1]

                session_iter[idx] = self.num_session_for_user[max_user_iter]
                session_start[idx] = self.session_offset[session_iter[idx]]
                session_end[idx] = self.session_offset[session_iter[idx] + 1]
