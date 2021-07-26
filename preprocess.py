import os
import argparse
import pandas as pd
from pandas import DataFrame
from typing import Tuple


def attach_sessions(df: DataFrame, threshold: int = 30*60):
    split_session = df["timestamp"].diff() > threshold

    new_user = df["user_id"] != df["user_id"].shift()
    new_session = new_user | split_session

    return df.assign(session_id=new_session.cumsum())


def split_by_timestamp(df: DataFrame, size: float = 0.8):
    timeorder = df.groupby("user_id")["timestamp"].rank(method="first", ascending=True)
    seen_cnts = df.groupby("user_id")["item_id"].transform("count")

    train = df[timeorder < seen_cnts * size]
    test = df[timeorder >= seen_cnts * size]

    test = test[test["item_id"].isin(train["item_id"].unique())]

    train, test = map(lambda df: df.reset_index(drop=True), (train, test))
    return train, test


def split_by_session(df: DataFrame, min_session_size: int = 2):
    last_sessions = df.groupby("user_id")["session_id"].transform("last")

    train = df[df["session_id"] != last_sessions]
    test = df[df["session_id"] == last_sessions]

    test = test[test["item_id"].isin(train["item_id"].unique())]

    good_sessions = test.groupby("session_id")["timestamp"].transform("count")
    test = test[good_sessions >= min_session_size]

    train, test = map(lambda df: df.reset_index(drop=True), (train, test))
    return train, test


def split_by_days(df: DataFrame, n_day: int, min_session_size: int = 2):
    DAY = 24 * 60 * 60

    end_time = df["timestamp"].max()
    test_start = end_time - n_day * DAY

    session_starts = df.groupby("session_id")["timestamp"].transform("min")

    train = df[session_starts < test_start]
    test = df[session_starts >= test_start]

    test = test[test["item_id"].isin(train["item_id"].unique())]

    good_sessions = test.groupby("session_id")["timestamp"].transform("count")
    test = test[good_sessions >= min_session_size]

    train, test = map(lambda df: df.reset_index(drop=True), (train, test))
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--data_dir", default="data", type=str)

    # data
    parser.add_argument("--raw_data", default="ml-1m.csv", type=str)
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--session_key", default="session_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)

    # preprocess
    parser.add_argument("--min_session_interval", default=30*60, type=int, help="1 hour = 60 min * 60 sec")
    parser.add_argument("--min_item_pop", default=10, type=int)
    parser.add_argument("--min_session_len", default=2, type=int)
    parser.add_argument("--session_per_user", default=(5, 199), type=Tuple[int])
    parser.add_argument("--split_session", default=1, type=int)

    # get the arguments
    args = parser.parse_args()  # with '.ipynb', use parser.parse_args([])

    interaction_dir = os.path.join(args.data_dir, args.raw_data)

    # load interaction data
    inter_df = pd.read_csv(interaction_dir)
    inter_df = attach_sessions(inter_df, args.min_session_interval)
    inter_df = inter_df.drop_duplicates(subset=[args.item_key, args.session_key], keep="first")

    item_pop = inter_df[args.item_key].value_counts()
    pop_items = item_pop.index[item_pop >= args.min_item_pop]
    dense_inter_df = inter_df[inter_df[args.item_key].isin(pop_items)]

    session_length = dense_inter_df.groupby(args.session_key)[args.time_key].transform("count")
    dense_inter_df = dense_inter_df[session_length >= args.min_session_len]

    sess_per_user = dense_inter_df.groupby(args.user_key)[args.session_key].transform("nunique")
    dense_inter_df = dense_inter_df[sess_per_user.between(*args.session_per_user)]

    train_sessions, test_sessions = split_by_session(dense_inter_df, args.split_session)
    train_sessions, valid_sessions = split_by_session(train_sessions, args.split_session)

    train_sessions.to_hdf("data/dense_train_sessions.hdf", "train")
    valid_sessions.to_hdf("data/dense_valid_sessions.hdf", "valid")
    test_sessions.to_hdf("data/dense_test_sessions.hdf", "test")
