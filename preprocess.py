import pandas as pd
from pandas import DataFrame


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
    SESSION_THRESHOLD = 2 * 60 * 60
    ITEM_POP_THRESHOLD = 20
    SESSION_LENGTH_THRESHOLD = 3

    # load interaction data
    inter_df = pd.read_csv("data/rb-10m.csv")
    inter_df = attach_sessions(inter_df, SESSION_THRESHOLD)
    inter_df = inter_df.drop_duplicates(subset=["item_id", "session_id"], keep="first")

    item_pop = inter_df["item_id"].value_counts()
    pop_items = item_pop.index[item_pop >= ITEM_POP_THRESHOLD]
    dense_inter_df = inter_df[inter_df["item_id"].isin(pop_items)]

    session_length = dense_inter_df.groupby("session_id")["timestamp"].transform("count")
    dense_inter_df = dense_inter_df[session_length >= SESSION_LENGTH_THRESHOLD]

    sess_per_user = dense_inter_df.groupby("user_id")["session_id"].transform("nunique")
    dense_inter_df = dense_inter_df[sess_per_user.between(5, 199)]

    train_sessions, test_sessions = split_by_session(dense_inter_df, 1)
    train_sessions, valid_sessions = split_by_session(train_sessions, 1)

    train_sessions.to_hdf("data/dense_train_sessions.hdf", "train")
    valid_sessions.to_hdf("data/dense_valid_sessions.hdf", "valid")
    test_sessions.to_hdf("data/dense_test_sessions.hdf", "test")