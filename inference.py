import os
import argparse
import torch
import pandas as pd

from src.dataset import Dataset
from src.model import HGRU4REC
from src.prediction import inference


def load_model(load_dir, device):
    model_state = torch.load(load_dir, map_location=device)
    model = HGRU4REC(
        device=device,
        **model_state["args"]
    )
    model.load_state_dict(model_state["model"])
    return model


def bootstrap(df, user_key="user_id", session_key="session_id", time_key="timestamp", bootstrap_len=2):
    user_sessions = (
        df
        .sort_values(by=[user_key, time_key])[[user_key, session_key]]
        .drop_duplicates()
    )

    session_order = (
        user_sessions
        .groupby(user_key, sort=False)
        .cumcount(ascending=False)
    )

    last_sessions = user_sessions[session_order < bootstrap_len][session_key]
    df = df[df[session_key].isin(last_sessions)]
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--model_name", default="HRNN-TOP1.pt", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--load_dir", default="model", type=str)

    # data
    parser.add_argument("--train_data", default="dense_train_sessions.hdf", type=str)
    parser.add_argument("--valid_data", default="dense_valid_sessions.hdf", type=str)
    parser.add_argument("--test_data", default="dense_test_sessions.hdf", type=str)
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--session_key", default="session_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)

    # inference
    parser.add_argument("--user_id", type=int, help="user id")
    parser.add_argument("--eval_k", default=25, type=int, help="how many items you recommend")
    parser.add_argument("--bootstrap_len", default=2, type=int, help="how many sessions will be bootstrapped")

    # get the arguments
    args = parser.parse_args()  # with '.ipynb', use parser.parse_args([])

    train_data_path = os.path.join(args.data_dir, args.train_data)
    valid_data_path = os.path.join(args.data_dir, args.valid_data)
    test_data_path = os.path.join(args.data_dir, args.test_data)
    load_dir = os.path.join(args.load_dir, args.model_name)

    device = torch.device("cpu")
    model = load_model(load_dir, device)

    train_df = pd.read_hdf(train_data_path, "train")
    valid_df = pd.read_hdf(valid_data_path, "valid")
    test_df = pd.read_hdf(test_data_path, "test")
    precede_df = pd.concat([train_df, valid_df], axis=0)

    precede_dataset = Dataset(precede_df)
    item_map = precede_dataset.item_map
    item_ids = precede_dataset.item_ids
    idx_map = {idx: id for id, idx in item_map.items()}

    test_df = test_df[test_df[args.item_key].isin(item_ids)]
    test_user_ids = test_df[args.user_key].unique()

    bootstrap_df = bootstrap(precede_df, bootstrap_len=args.bootstrap_len)
    bootstrap_df = bootstrap_df[bootstrap_df[args.user_key].isin(test_user_ids)]
    bootstrap_df = pd.concat([bootstrap_df, test_df], axis=0)

    return inference(args.user_id, model, bootstrap_df, device, item_map, idx_map, eval_k=args.eval_k)


if __name__ == "__main__":
    main()
