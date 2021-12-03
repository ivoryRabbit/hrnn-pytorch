import os
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from argparser import set_env


def read_raw_data(data_dir: str, user_key: str, item_key: str, time_key: str) -> DataFrame:
    df = pd.read_csv(data_dir, converters={"timestamp": lambda t: int(t)})
    name_map = {
        user_key: "user_id",
        item_key: "item_id",
        time_key: "timestamp",
    }
    return df.rename(columns=name_map).filter(items=["user_id", "item_id", "timestamp"])


def attach_session(df: DataFrame, threshold: int = 30 * 60) -> DataFrame:
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    split_user = df["user_id"] != df["user_id"].shift()
    split_time = df["timestamp"].diff() > threshold

    session = split_user | split_time
    return df.assign(session_id=session.cumsum())


def drop_duplicate(df: DataFrame) -> DataFrame:
    return df.drop_duplicates(subset=["user_id", "item_id", "session_id"], keep="last")


def drop_sparse_item(df: DataFrame, min_item_pop: int = 5) -> DataFrame:
    item_pop = df.groupby("item_id")["timestamp"].count()

    fine_item = item_pop.index[item_pop >= min_item_pop]
    return df[df["item_id"].isin(fine_item)]


def drop_sparse_session(df: DataFrame, min_session_size: int = 3) -> DataFrame:
    session_size = df.groupby("session_id")["item_id"].count()

    fine_session = session_size.index[session_size >= min_session_size]
    return df[df["session_id"].isin(fine_session)]


def drop_sparse_user(df: DataFrame, min_session_len: int = 5) -> DataFrame:
    session_len = df.groupby("user_id")["session_id"].count()

    fine_user = session_len.index[session_len >= min_session_len]
    return df[df["user_id"].isin(fine_user)]


def drop_outlier(df: DataFrame, history_len_percentile: Tuple[float, float]) -> DataFrame:
    user_history_len = inter_df.groupby("user_id")["item_id"].count()

    min_len, max_len = map(user_history_len.quantile, history_len_percentile)

    fine_user = user_history_len.index[user_history_len.between(min_len, max_len)]
    return df[df["user_id"].isin(fine_user)]


def split_by_session(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    last_session = df.groupby("user_id")["session_id"].transform("last")

    train = df[df["session_id"] != last_session]
    test = df[df["session_id"] == last_session]

    session_len = train.groupby("session_id")["timestamp"].count()
    trainable_session = session_len.index[session_len > 1]
    train = train[train["session_id"].isin(trainable_session)]

    predictable_item = train["item_id"].unique()
    test = test[test["item_id"].isin(predictable_item)]

    train, test = map(lambda df: df.reset_index(drop=True), (train, test))
    return train, test


def attach_item_idx(df: DataFrame) -> DataFrame:
    item_ids = df["item_id"].unique()
    to_idx_map = {item: idx for idx, item in enumerate(item_ids)}
    return (
        df
        .assign(item_idx=lambda df: df["item_id"].map(to_idx_map))
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    args = set_env()

    # load data
    inter_df = read_raw_data(
        os.environ["interaction_dir"],
        user_key=args.user_key, item_key=args.item_key, time_key=args.time_key,
    )

    # refine session data
    inter_df = attach_session(inter_df, args.min_session_interval)
    inter_df = drop_duplicate(inter_df)

    # drop sparse data
    parsed_inter_df = drop_sparse_item(inter_df, args.min_item_pop)
    parsed_inter_df = drop_sparse_session(parsed_inter_df, args.min_session_size)
    parsed_inter_df = drop_sparse_user(parsed_inter_df, args.min_session_len)
    # parsed_inter_df = drop_outlier(inter_df, args.history_len_percentile)

    # split train, valid, test (leave-one-out w.r.t. session)
    train_df, test_df = split_by_session(parsed_inter_df)
    train_df, valid_df = split_by_session(train_df)

    try:
        item_meta_df = read_raw_data(
            os.environ["item_meta_dir"],
            user_key=args.user_key, item_key=args.item_key, time_key=args.time_key,
        )
    except KeyError:
        item_meta_df = pd.DataFrame({"item_id": inter_df["item_id"].unique()})

    # create trainable item meta data
    item_for_train_df = (
        train_df
        .groupby("item_id", as_index=False)[["timestamp"]]
        .agg("max")
    )
    item_for_train_df = (
        item_for_train_df
        .merge(item_meta_df, on="item_id", how="inner")
        .sort_values(by="timestamp")
        .drop(columns="timestamp")
    )
    item_for_train_df = attach_item_idx(item_for_train_df)

    # get precede data with sessions
    trainable_items = item_for_train_df["item_id"].unique()
    inter_df = inter_df[inter_df["item_id"].isin(trainable_items)]

    # save data
    inter_df.to_csv(os.environ["inter_dir"], index=False)
    train_df.to_csv(os.environ["train_dir"], index=False)
    valid_df.to_csv(os.environ["valid_dir"], index=False)
    test_df.to_csv(os.environ["test_dir"], index=False)
    item_for_train_df.to_csv(os.environ["item_dir"], index=False)
