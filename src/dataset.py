import numpy as np
import torch


class Sampler(object):
    def __init__(self, df, n_samples, item_key="item_id", sample_alpha=0.75, sample_store=10000):
        self.df = df
        self.n_samples = n_samples
        self.item_key = item_key
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
        """
        Calculate the distribution P(w_i) of negative sampling
        """
        prob = self.df[self.item_key].value_counts() ** self.sample_alpha
        return prob.cumsum() / prob.sum()


class Dataset(object):
    def __init__(
        self,
        df,
        item_map=None,
        user_key="user_id",
        session_key="session_id",
        item_key="item_id",
        time_key="timestamp",
    ):
        self.df = df
        self.user_key = user_key
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        # Add colummn item index to dataframe
        self.item_map = self.refine_item_map(item_map)
        self.attach_item_indices()

        # Sort dataframe
        self.df.sort_values([user_key, session_key, time_key], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def refine_item_map(self, item_map):
        if item_map is None:
            item_map = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        return item_map

    def attach_item_indices(self):
        self.df = self.df.assign(item_idx=lambda df: df[self.item_key].map(self.item_map))

    @property
    def session_offsets(self):
        return np.r_[0, self.df.groupby(self.session_key, sort=False).size().cumsum().values]

    @property
    def num_sessions_each_user(self):
        return np.r_[0, self.df.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum().values]

    @property
    def user_idx_arr(self):
        return np.arange(self.df[self.user_key].nunique())

    @property
    def session_idx_arr(self):
        return np.arange(self.df[self.session_key].nunique())

    @property
    def item_ids(self):
        return self.df[self.item_key].unique()


class DataLoader(object):
    def __init__(self, dataset: Dataset, batch_size=50, n_samples=-1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = n_samples

    def __iter__(self):
        """
        Yields:
        """
        df = self.dataset.df
        neg_sampler = Sampler(df, self.n_samples)

        session_offsets = self.dataset.session_offsets
        num_sessions_each_user = self.dataset.num_sessions_each_user
        user_offsets = session_offsets[num_sessions_each_user]

        user_idx_arr = self.dataset.user_idx_arr
        user_iters = np.arange(self.batch_size)
        user_max_iter = user_iters.max()

        user_start = user_offsets[user_idx_arr[user_iters]]
        user_end = user_offsets[user_idx_arr[user_iters] + 1]

        session_iters = num_sessions_each_user[user_iters]

        session_start = session_offsets[session_iters]
        session_end = session_offsets[session_iters + 1]

        session_change_idx = []
        user_change_idx = []
        finished = False

        while not finished:
            session_interval = (session_end - session_start).min()

            for i in range(session_interval - 1):
                input_idx = df["item_idx"].values[session_start + i]
                output_idx = df["item_idx"].values[session_start + i + 1]

                if self.n_samples > 0:
                    output_idx = np.hstack([output_idx, next(neg_sampler)])

                input = torch.LongTensor(input_idx)
                output = torch.LongTensor(output_idx)

                yield input, output, session_change_idx, user_change_idx

            session_start += session_interval - 1
            session_change_idx = np.arange(len(session_iters))[(session_end - session_start) <= 1]

            for idx in session_change_idx:
                session_iters[idx] += 1
                if session_iters[idx] + 1 >= len(session_offsets):
                    finished = True
                    break

                session_start[idx] = session_offsets[session_iters[idx]]
                session_end[idx] = session_offsets[session_iters[idx] + 1]

            user_change_idx = np.arange(len(user_iters))[(user_end - session_start <= 0)]

            for idx in user_change_idx:
                user_max_iter += 1
                if user_max_iter + 1 >= len(user_offsets):
                    finished = True
                    break

                user_iters[idx] = user_max_iter
                user_start[idx] = user_offsets[user_max_iter]
                user_end[idx] = user_offsets[user_max_iter + 1]

                session_iters[idx] = num_sessions_each_user[user_max_iter]
                session_start[idx] = session_offsets[session_iters[idx]]
                session_end[idx] = session_offsets[session_iters[idx] + 1]
