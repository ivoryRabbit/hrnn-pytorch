import os
import pandas as pd
import torch

from src.dataset import DataLoader, DenseIndexing, Masking, ToTensor
from src.model import HGRU4REC
from src.optimizer import Optimizer
from src.loss_function import LossFunction
from src.metric import Metric
from src.trainer import Trainer
from src.callback import EarlyStopping
from argparser import set_env


if __name__ == "__main__":
    args = set_env()

    # check gpu environment
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load data
    train_df = pd.read_csv(os.environ["train_dir"])
    valid_df = pd.read_csv(os.environ["valid_dir"])

    item_df = pd.read_csv(os.environ["item_dir"])
    input_size = output_size = len(item_df)

    transforms = [
        DenseIndexing(item_df),
        Masking(),
        ToTensor(device),
    ]
    train_loader = DataLoader(args, train_df, transforms=transforms)
    valid_loader = DataLoader(args, valid_df, transforms=transforms)

    model = HGRU4REC(
        device=device,
        input_size=input_size,
        output_size=output_size,
        hidden_dim=args.hidden_dim,
        dropout_init=args.dropout_init,
        dropout_user=args.dropout_user,
        dropout_session=args.dropout_session,
    )

    optimizer = Optimizer(
        args,
        params=model.parameters(),
        optimizer_type=args.optimizer_type,
    )

    loss_function = LossFunction(loss_type=args.loss_type)

    metric = Metric(device, eval_k=args.eval_k)

    early_stopping = EarlyStopping(args, checkpoint_dir=os.environ["checkpoint_dir"])

    trainer = Trainer(
        args, model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        metric=metric,
        early_stopping=early_stopping,
    )

    best_model, train_losses, eval_losses = trainer.train(args.n_epochs)
    best_model.save(save_dir=os.environ["save_dir"])
