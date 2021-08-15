import argparse
import os
from typing import Optional, Tuple


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--interaction", default="ml-1m.csv", type=str)
    parser.add_argument("--item_meta", default=None, type=Optional[str])
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)

    # preprocess
    parser.add_argument("--min_session_interval", default=60*60, type=int, help="1hour=60*60sec")
    parser.add_argument("--min_item_pop", default=10, type=int)
    parser.add_argument("--min_session_len", default=5, type=int)
    parser.add_argument("--session_per_user", default=(5, 99), type=Tuple[int])
    parser.add_argument("--leave_out_session", default=1, type=int)

    # model
    parser.add_argument("--model_name", default="HRNN-TOP1.pt", type=str)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--n_layers", default=1, type=int,
                        help="having implemented, multiple layers do not improve performance")
    parser.add_argument("--hidden_act", default="tanh", type=str)
    parser.add_argument("--final_act", default="tanh", type=str)
    parser.add_argument("--dropout_init", default=0.1, type=float)
    parser.add_argument("--dropout_user", default=0.1, type=float)
    parser.add_argument("--dropout_session", default=0.1, type=float)
    parser.add_argument("--fft_all", default=False, type=bool, help="feed forward to all")
    parser.add_argument("--sigma", default=-1, type=float,
                        help="init weight, -1: range [-sigma, sigma], -2: range [0, sigma]")
    parser.add_argument("--seed", default=777, type=int, help="seed for random initialization")

    # learning parameters
    parser.add_argument("--n_samples", default=-1, type=int)
    parser.add_argument("--loss_type", default="TOP1", type=str,
                        help="TOP1, BPR, TOP1-max, BPR-max")
    parser.add_argument("--optimizer_type", default="Adagrad", type=str,
                        help="Adagrad, Adam, RMSProp, SGD")
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.1, type=float)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--patience", default=5, type=int, help="early stopping patience")
    parser.add_argument("--delta", default=0.0, type=float, help="early stopping threshold")
    parser.add_argument("--verbose", default=True, type=bool)

    # inference
    parser.add_argument("--eval_k", default=25, type=int, help="how many items you recommend")
    parser.add_argument("--user_id", type=int, help="user id")

    args = parser.parse_args()
    return args


def set_env(data_root="data", save_root="trained", checkpoint_root="checkpoint"):
    args = get_args()

    # raw data
    os.environ["interaction_dir"] = os.path.join(data_root, args.interaction)
    if args.item_meta:
        os.environ["item_meta_dir"] = os.path.join(data_root, args.item_meta)

    # data
    os.environ["train_dir"] = os.path.join(data_root, "train_data.hdf")
    os.environ["valid_dir"] = os.path.join(data_root, "valid_data.hdf")
    os.environ["test_dir"] = os.path.join(data_root, "test_data.hdf")
    os.environ["item_dir"] = os.path.join(data_root, "item_for_train.csv")

    # pre-trained
    os.environ["save_dir"] = os.path.join(save_root, args.model_name)
    os.environ["checkpoint_dir"] = os.path.join(checkpoint_root, args.model_name)

    return args
