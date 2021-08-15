import os
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from setup import set_env


def read_raw_data(data_dir: str, user_key: str, item_key: str, time_key: str) -> DataFrame:
    name_map = {
        user_key: "user_id",
        item_key: "item_id",
        time_key: "timestamp",
    }
    return pd.read_csv(data_dir).rename(columns=name_map)


def attach_sessions(df: DataFrame, threshold: int = 30*60) -> DataFrame:
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    split_session = df["timestamp"].diff() > threshold

    new_user = df["user_id"] != df["user_id"].shift()
    new_session = new_user | split_session

    return (
        df
        .assign(session_id=new_session.cumsum())
        .drop_duplicates(subset=["item_id", "session_id"], keep="first")
    )


def attach_item_idx(df: DataFrame) -> DataFrame:
    item_ids = df["item_id"].unique()
    to_idx_map = {Id: idx for idx, Id in enumerate(item_ids)}
    return df.assign(item_idx = lambda df: df["item_id"].map(to_idx_map))


def split_by_session(df: DataFrame, min_session_size: int = 2) -> Tuple[DataFrame, DataFrame]:
    last_sessions = df.groupby("user_id")["session_id"].transform("last")

    train = df[df["session_id"] != last_sessions]
    test = df[df["session_id"] == last_sessions]

    test = test[test["item_id"].isin(train["item_id"].unique())]

    good_sessions = test.groupby("session_id")["timestamp"].transform("count")
    test = test[good_sessions >= min_session_size]
    return train, test


def split_by_timestamp(df: DataFrame, size: float = 0.8) -> Tuple[DataFrame, DataFrame]:
    timeorder = df.groupby("user_id")["timestamp"].rank(method="first", ascending=True)
    seen_cnts = df.groupby("user_id")["item_id"].transform("count")

    train = df[timeorder < seen_cnts * size]
    test = df[timeorder >= seen_cnts * size]

    test = test[test["item_id"].isin(train["item_id"].unique())]
    return train, test


def split_by_days(df: DataFrame, n_day: int, min_session_size: int = 2) -> Tuple[DataFrame, DataFrame]:
    DAY = 24 * 60 * 60

    end_time = df["timestamp"].max()
    test_start = end_time - n_day * DAY

    session_starts = df.groupby("session_id")["timestamp"].transform("min")

    train = df[session_starts < test_start]
    test = df[session_starts >= test_start]

    test = test[test["item_id"].isin(train["item_id"].unique())]

    good_sessions = test.groupby("session_id")["timestamp"].transform("count")
    test = test[good_sessions >= min_session_size]
    return train, test


def drop_sparse_item(df: DataFrame, min_item_pop: int = 10) -> DataFrame:
    item_pop = df.groupby("item_id")["timestamp"].transform("nunique")
    return df[item_pop >= min_item_pop]


def drop_sparse_session(df: DataFrame, min_session_len: int = 5) -> DataFrame:
    session_len = df.groupby("session_id")["timestamp"].transform("nunique")
    return df[session_len >= min_session_len]


def drop_sparse_user(df: DataFrame, session_per_user: Tuple[int]) -> DataFrame:
    sess_per_user = df.groupby("user_id")["session_id"].transform("nunique")
    return df[sess_per_user.between(*session_per_user)]


if __name__ == "__main__":
    args = set_env()

    # load data
    inter_df = read_raw_data(
        os.environ["interaction_dir"],
        user_key=args.user_key, item_key=args.item_key, time_key=args.time_key,
    )

    try:
        item_df = read_raw_data(
            os.environ["item_meta_dir"],
            user_key=args.user_key, item_key=args.item_key, time_key=args.time_key,
        )
    except KeyError:
        item_df = pd.DataFrame({"item_id": inter_df["item_id"].unique()})

    # drop sparse data
    inter_df = attach_sessions(inter_df, args.min_session_interval)
    inter_df = drop_sparse_item(inter_df, args.min_item_pop)
    inter_df = drop_sparse_session(inter_df, args.min_session_len)
    inter_df = drop_sparse_user(inter_df, args.session_per_user)

    # split train, valid, test (leave-one-out w.r.t. session)
    train_df, test_df = split_by_session(inter_df, args.leave_out_session)
    train_df, valid_df = split_by_session(train_df, args.leave_out_session)

    # drop non-exist items
    items = inter_df.item_id.unique()
    item_df = item_df[item_df["item_id"].isin(items)].reset_index(drop=True)
    item_df = attach_item_idx(item_df)

    # save data
    train_df.to_hdf(os.environ["train_dir"], "train")
    valid_df.to_hdf(os.environ["valid_dir"], "valid")
    test_df.to_hdf(os.environ["test_dir"], "test")
    item_df.to_csv(os.environ["item_dir"], index=False)
