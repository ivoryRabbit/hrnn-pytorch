import torch


class SessionDataLoader(object):
    def __init__(self, df):
        self.df = df
        self.batch_size = 1

    def __iter__(self):
        prev_session = None
        for session, item in self.df.values:
            mask = []
            if prev_session and prev_session != session:
                mask.append(0)
            yield torch.LongTensor([item]), mask

            prev_session = session


def predict_batch(dataset, model, batch_size, loss_function, eval_k, device):
    import numpy as np
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


def inference(user_id, df, model, device):
    user_hist = df[df["user_id"] == user_id][["session_id", "item_idx"]]
    user_dataloader = SessionDataLoader(user_hist)

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
    return torch.flatten(score)
