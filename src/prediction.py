import numpy as np
import torch


class SessionDataset(object):
    def __init__(
        self,
        df,
        item_map=None,
        session_key="session_id",
        item_key="item_id",
        time_key="timestamp",
    ):
        self.df = df
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        self.item_map = self.refine_item_map(item_map)
        self.attach_item_indices()

        self.df.sort_values([session_key, time_key], inplace=True)
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
    def session_idx_arr(self):
        return np.arange(self.df[self.session_key].nunique())

    @property
    def item_ids(self):
        return self.df[self.item_key].unique()


class SessionDataLoader(object):
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        df = self.dataset.df
        session_offsets = self.dataset.session_offsets
        session_idx_arr = self.dataset.session_idx_arr
        batch_size = min(self.batch_size, len(session_idx_arr))

        iters = np.arange(batch_size)
        max_iter = iters.max()
        start = session_offsets[session_idx_arr[iters]]
        end = session_offsets[session_idx_arr[iters] + 1]
        mask = []
        finished = False

        while not finished:
            min_interval = (end - start).min()
            for i in range(min_interval - 1):
                idx_input = df["item_idx"].values[start + i]
                input = torch.LongTensor(idx_input)
                yield input, mask

            start += (min_interval - 1)
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                max_iter += 1
                if max_iter >= len(session_offsets) - 1:
                    finished = True
                    break
                iters[idx] = max_iter
                start[idx] = session_offsets[session_idx_arr[max_iter]]
                end[idx] = session_offsets[session_idx_arr[max_iter] + 1]


def predict_batch(dataset, model, batch_size, loss_function, eval_k, device):
    from tqdm import tqdm
    from src.dataset import DataLoader
    from src.evaluation import evaluate

    model.eval()
    losses = []
    metrics = {}

    data_loader = DataLoader(dataset, batch_size, -1)

    user_repr = model.init_hidden(batch_size)
    session_repr = model.init_hidden(batch_size)

    with torch.no_grad():
        for input, output, session_start, user_start in tqdm(data_loader):
            input = input.to(device)
            output = output.to(device)

            session_mask = model.get_mask(session_start, batch_size)
            user_mask = model.get_mask(user_start, batch_size)

            score, session_repr, user_repr = model(input, session_repr, session_mask, user_repr,
                                                   user_mask)
            sampled_score = score[:, output.view(-1)]

            loss = loss_function(sampled_score)
            eval_metrics = evaluate(score, output, k=eval_k)

            losses.append(loss.item())

            for metric, value in eval_metrics.items():
                metrics[metric] = metrics.get(metric, []) + [value]

    mean_losses = np.mean(losses)
    mean_metrics = {metric: np.mean(values) for metric, values in metrics.items()}

    return mean_losses, mean_metrics


def inference(user_id, model, df, device, item_map, idx_map, user_key="user_id", eval_k=20):
    user_info = df[df[user_key] == user_id]
    user_dataset = SessionDataset(user_info, item_map)
    user_dataloader = SessionDataLoader(user_dataset, 1)

    model.to(device)
    model.eval()
    user_repr = model.init_hidden(1)
    session_repr = model.init_hidden(1)

    with torch.no_grad():
        for item, session_start in user_dataloader:
            item = item.to(device)

            session_mask = model.get_mask(session_start, 1)
            user_mask = model.get_mask([], 1)

            score, session_repr, user_repr = model(
                item, session_repr, session_mask, user_repr, user_mask
            )

    _, indices = torch.topk(score[-1], eval_k)
    indices = indices.cpu().numpy()
    return np.vectorize(idx_map.get)(indices)
