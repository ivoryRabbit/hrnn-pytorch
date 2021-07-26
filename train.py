import os
import argparse
import numpy as np
import pandas as pd
import torch

from src.dataset import Dataset
from src.model import HGRU4REC
from src.optimizer import Optimizer
from src.loss_function import LossFunction
from src.trainer import EarlyStopping, Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--model_name", default="HRNN-TOP1.pt", type=str)
    parser.add_argument("--load_model", default=None,  type=str, help="use pre-trained model")
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--save_dir", default="model", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoint", type=str)

    # data
    parser.add_argument("--train_data", default="dense_train_sessions.hdf", type=str)
    parser.add_argument("--valid_data", default="dense_valid_sessions.hdf", type=str)
    parser.add_argument("--test_data", default="dense_test_sessions.hdf", type=str)
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--session_key", default="session_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)
    parser.add_argument("--value_key", default="event_value", type=str)
    parser.add_argument("--n_samples", default=-1, type=int)

    # learning
    parser.add_argument("--loss_type", default="TOP1", type=str, help="TOP1, BPR, TOP1-max, BPR-max")
    parser.add_argument("--optimizer_type", default="Adagrad", type=str, help="Adagrad, Adam, RMSProp, SGD")
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.1, type=float)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--patience", default=5, type=int, help="early stopping patience")

    # model
    parser.add_argument("--embedding_dim", default=-1, type=int, help="if positive, using item embedding")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--n_layers", default=1, type=int, help="try to implement, multiple layers do not improve performance")
    parser.add_argument("--hidden_act", default="tanh", type=str)
    parser.add_argument("--final_act", default="tanh", type=str)
    parser.add_argument("--dropout_init", default=0.1, type=float)
    parser.add_argument("--dropout_user", default=0.1, type=float)
    parser.add_argument("--dropout_session", default=0.1, type=float)
    parser.add_argument("--fft_all", default=False, type=bool, help="feed forward to all")
    parser.add_argument("--sigma", default=-1, type=float, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
    parser.add_argument("--seed", default=777, type=int, help="seed for random initialization")

    # inference
    parser.add_argument("--eval_k", default=25, type=int, help="how many items you recommend")

    # get the arguments
    # with '.ipynb', use parser.parse_args([])
    args = parser.parse_args()

    # check gpu environment
    use_cuda = torch.cuda.is_available()

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # data
    train_data_path = os.path.join(args.data_dir, args.train_data)
    valid_data_path = os.path.join(args.data_dir, args.valid_data)
    test_data_path = os.path.join(args.data_dir, args.test_data)
    save_dir = os.path.join(args.save_dir, args.model_name)

    # check gpu environment
    device = torch.device("cuda" if use_cuda else "cpu")

    train_df = pd.read_hdf(train_data_path, "train")
    valid_df = pd.read_hdf(valid_data_path, "valid")

    train_dataset = Dataset(train_df)
    item_map = train_dataset.item_map
    input_size = output_size = len(item_map)

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
        embedding_dim=args.embedding_dim,
    )

    optimizer = Optimizer(
        model.parameters(),
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        eps=args.eps,
    )

    loss_function = LossFunction(loss_type=args.loss_type)

    early_stopping = EarlyStopping(args, patience=args.patience, verbose=True)

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
    best_model.save(save_dir)
