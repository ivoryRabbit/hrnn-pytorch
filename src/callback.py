import numpy as np
import torch


class EarlyStopping(object):
    def __init__(self, args, checkpoint_dir: str, mode="min"):
        self.patience = args.patience
        self.delta = args.delta
        self.checkpoint_dir = checkpoint_dir

        assert mode in ("min", "max"), "mode should be 'min' or 'max'"
        self.mode = mode
        self.best_eval_loss = np.Inf if mode == "min" else -np.Inf

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, eval_loss, checkpoint):
        score = -eval_loss if self.mode == "min" else eval_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(eval_loss, checkpoint)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"Validation loss decreased: {self.best_eval_loss:.5f} --> {eval_loss:.5f}")
            self.best_score = score
            self.save_checkpoint(eval_loss, checkpoint)
            self.counter = 0
        return self.checkpoint_dir

    def save_checkpoint(self, eval_loss, checkpoint):
        print("Save trained model...")

        torch.save(checkpoint, self.checkpoint_dir)
        self.best_eval_loss = eval_loss
