import numpy as np
import torch
import time
from tqdm import tqdm


class Trainer(object):
    def __init__(
        self,
        args,
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_function,
        metric,
        early_stopping,
    ):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metric = metric
        self.early_stopping = early_stopping

    def train(self, n_epochs):
        self.model.init_model(self.args.sigma)

        start_time = time.time()
        checkpoint_dir = None
        train_losses, eval_losses = [], []

        for epoch in range(1, n_epochs + 1):
            st = time.time()
            print(f"Start Epoch #{epoch}")

            train_loss = self.train_step()
            train_losses.append(train_loss)

            eval_loss, metrics = self.valid_step()
            eval_losses.append(eval_loss)

            epoch_len = len(str(n_epochs))
            print(
                f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] " +
                f"--train_loss: {train_loss:.5f} " +
                f"--valid_loss: {eval_loss:.5f} " +
                "".join([
                    f"--{metric}: {value:.5f} "
                    for metric, value in metrics.items()
                ]) +
                f"--time: {time.time() - st:0.1f} sec"
            )

            checkpoint = dict(
                model=self.model.state_dict(),
                epoch=epoch,
                optimizer=self.optimizer,
                loss=eval_loss,
            )
            checkpoint.update(metrics)

            checkpoint_dir = self.early_stopping(eval_loss, checkpoint)
            if self.early_stopping.early_stop:
                print("Early stopped")
                break

        if checkpoint_dir is not None:
            checkpoint = torch.load(checkpoint_dir)
            self.model.load_state_dict(checkpoint["model"])

        print(f"time: {time.time() - start_time:0.1f} sec")
        return self.model, train_losses, eval_losses

    def train_step(self):
        self.model.train()
        losses = []

        user_repr = self.model.init_hidden(self.args.batch_size)
        session_repr = self.model.init_hidden(self.args.batch_size)

        for samples in tqdm(self.train_loader, miniters=1000):
            self.optimizer.zero_grad()
            inputs = samples["inputs"]
            targets = samples["targets"]
            session_mask = samples["session_change"]
            user_mask = samples["session_change"]

            score, next_session_repr, next_user_repr = self.model(
                inputs, session_repr, session_mask, user_repr, user_mask
            )
            sampled_score = score[:, targets.view(-1)]

            loss = self.loss_function(sampled_score)
            loss.backward()

            losses.append(loss.item())
            self.optimizer.step()

            session_repr = next_session_repr.detach()
            user_repr = next_user_repr.detach()

        mean_losses = np.mean(losses)
        return mean_losses

    def valid_step(self):
        self.model.eval()
        losses = []
        metrics = {}

        user_repr = self.model.init_hidden(self.args.batch_size)
        session_repr = self.model.init_hidden(self.args.batch_size)

        with torch.no_grad():
            for samples in tqdm(self.valid_loader, miniters=1000):
                inputs = samples["inputs"]
                targets = samples["targets"]
                session_mask = samples["session_change"]
                user_mask = samples["session_change"]

                score, session_repr, user_repr = self.model(
                    inputs, session_repr, session_mask, user_repr, user_mask
                )
                sampled_score = score[:, targets.view(-1)]

                loss = self.loss_function(sampled_score)
                eval_metrics = self.metric(score, targets)

                losses.append(loss.item())

                for metric, value in eval_metrics.items():
                    metrics[metric] = metrics.get(metric, []) + [value]

        mean_loss = np.mean(losses)
        mean_metrics = {metric: np.mean(values) for metric, values in metrics.items()}

        return mean_loss, mean_metrics
