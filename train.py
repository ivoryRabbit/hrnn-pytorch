import os
import pandas as pd
import torch

from src.dataset import Dataset
from src.model import HGRU4REC
from src.optimizer import Optimizer
from src.loss_function import LossFunction
from src.trainer import EarlyStopping, Trainer
from setup import set_env


if __name__ == "__main__":
    args = set_env()

    # check gpu environment
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load data
    train_df = pd.read_hdf(os.environ["train_dir"], "train")
    valid_df = pd.read_hdf(os.environ["valid_dir"], "valid")

    item_df = pd.read_csv(os.environ["item_dir"])
    item_map = {Id: idx for Id, idx in item_df[["item_id", "item_idx"]].values}
    input_size = output_size = len(item_map)

    train_dataset = Dataset(train_df, item_map)
    valid_dataset = Dataset(valid_df, item_map)

    model = HGRU4REC(
        device=device,
        input_size=input_size,
        output_size=output_size,
        hidden_dim=args.hidden_dim,
        final_act=args.final_act,
        hidden_act=args.hidden_act,
        dropout_init=args.dropout_init,
        dropout_user=args.dropout_user,
        dropout_session=args.dropout_session,
    )

    optimizer = Optimizer(
        args,
        model.parameters(),
        optimizer_type=args.optimizer_type,
    )

    loss_function = LossFunction(loss_type=args.loss_type)

    early_stopping = EarlyStopping(args, checkpoint_dir=os.environ["checkpoint_dir"])

    trainer = Trainer(
        args,
        model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        optimizer=optimizer,
        loss_function=loss_function,
        early_stopping=early_stopping,
        device=device,
    )

    best_model, train_losses, eval_losses = trainer.train(args.n_epochs)
    best_model.save(save_dir=os.environ["save_dir"])
