import os
import numpy as np
import torch
import time
from tqdm import tqdm

from dataset import DataLoader
from evaluation import evaluate


class EarlyStopping(object):
    def __init__(self, args, patience=7, verbose=False, delta=0, mode="min"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        assert mode in ("min", "max"), "mode should be 'min' or 'max'"
        self.mode = mode

        self.best_eval_loss = np.Inf if mode == "min" else -np.Inf
        self.delta = delta
        self.checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)

    def __call__(self, eval_loss, checkpoint):
        if self.mode == "min":
            score = -eval_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(eval_loss, checkpoint)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(eval_loss, checkpoint)
            self.counter = 0
        return self.checkpoint_path

    def save_checkpoint(self, eval_loss, checkpoint):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_eval_loss:.5f} --> {eval_loss:.5f})")
            print("Saving model...")

        torch.save(checkpoint, self.checkpoint_path)
        self.best_eval_loss = eval_loss


class Trainer(object):
    def __init__(self, args, model, train_data, valid_data, optimizer, loss_function,
                 early_stopping, device):
        self.args = args
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.early_stopping = early_stopping
        self.device = device
        self.batch_size = args.batch_size
        self.eval_k = args.eval_k

    def train(self, n_epochs):
        self.start_time = time.time()
        self.model.init_model(self.args.sigma)

        train_losses, eval_losses = [], []

        for epoch in range(1, n_epochs + 1):
            st = time.time()
            print("Start Epoch #", epoch)

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

            checkpoint_path = self.early_stopping(eval_loss, checkpoint)
            if self.early_stopping.early_stop:
                print("Early stopped")
                break

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        print(f"time: {time.time() - self.start_time:0.1f} sec")
        return self.model, train_losses, eval_losses

    def train_step(self):
        self.model.train()
        losses = []

        train_loader = DataLoader(self.train_data, self.batch_size, -1)

        user_repr = self.model.init_hidden(self.batch_size)
        session_repr = self.model.init_hidden(self.batch_size)

        for input, output, session_start, user_start in tqdm(train_loader, miniters=1000):
            self.optimizer.zero_grad()
            input = input.to(self.device)
            output = output.to(self.device)

            session_mask = self.model.get_mask(session_start, self.batch_size)
            user_mask = self.model.get_mask(user_start, self.batch_size)

            score, next_session_repr, next_user_repr = self.model(input, session_repr, session_mask,
                                                                  user_repr, user_mask)
            sampled_score = score[:, output.view(-1)]

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

        valid_loader = DataLoader(self.valid_data, self.batch_size, -1)

        user_repr = self.model.init_hidden(self.batch_size)
        session_repr = self.model.init_hidden(self.batch_size)

        with torch.no_grad():
            for input, output, session_start, user_start in tqdm(valid_loader, miniters=1000):
                input = input.to(self.device)
                output = output.to(self.device)

                session_mask = self.model.get_mask(session_start, self.batch_size)
                user_mask = self.model.get_mask(user_start, self.batch_size)

                score, session_repr, user_repr = self.model(input, session_repr, session_mask,
                                                            user_repr, user_mask)
                sampled_score = score[:, output.view(-1)]

                loss = self.loss_function(sampled_score)
                eval_metrics = evaluate(score, output, k=self.eval_k)

                losses.append(loss.item())

                for metric, value in eval_metrics.items():
                    metrics[metric] = metrics.get(metric, []) + [value]

        mean_loss = np.mean(losses)
        mean_metrics = {metric: np.mean(values) for metric, values in metrics.items()}

        return mean_loss, mean_metrics