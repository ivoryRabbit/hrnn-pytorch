import os
import torch
import pandas as pd
import numpy as np

from src.model import HGRU4REC
from src.prediction import inference
from src.coldstart import ColdStart
from src.filter import Filter


class Recommend(object):
    def __init__(self, bootstrap_len=2):
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)

        self.device = torch.device("cpu")
        self.model = self._load_model()
        self.item_df = self._get_item_meta()
        self.to_idx_map = {Id: idx for Id, idx in self.item_df.values}
        self.to_id_map = {idx: Id for Id, idx in self.item_df.values}
        self.precede_df = self._get_precede()
        self.bootstrapped_test_df = self._get_bootstrapped_test(bootstrap_len)
        self.cold_start = ColdStart(self.precede_df)
        self.filter = Filter(self.precede_df, self.item_df)

    def _load_model(self):
        model_state = torch.load(os.environ["model_dir"], map_location=self.device)
        model = HGRU4REC(
            device=self.device,
            **model_state["args"]
        )
        model.load_state_dict(model_state["model"])
        return model

    def _get_precede(self):
        train_df = pd.read_hdf(os.environ["train_dir"], "train")
        valid_df = pd.read_hdf(os.environ["valid_dir"], "valid")
        return pd.concat([train_df, valid_df], axis=0)

    @staticmethod
    def _get_item_meta():
        df = pd.read_csv(os.environ["item_dir"])
        return df[["item_id", "item_idx"]]

    @staticmethod
    def _bootstrap(df, bootstrap_len=2):
        user_sessions = (
            df
            .sort_values(by=["user_id", "timestamp"])[["user_id", "session_id"]]
            .drop_duplicates()
        )

        session_order = (
            user_sessions
            .groupby("user_id", sort=False)
            .cumcount(ascending=False)
        )

        last_sessions = user_sessions[session_order < bootstrap_len]["session_id"]
        return df[df["session_id"].isin(last_sessions)]

    def _get_bootstrapped_test(self, bootstrap_len=2):
        test_df = pd.read_hdf(os.environ["test_dir"], "test")

        item_ids = self.precede_df["user_id"].unique()
        test_df = test_df[test_df["item_id"].isin(item_ids)]

        test_user_ids = test_df["user_id"].unique()
        precede_df = self.precede_df[self.precede_df["user_id"].isin(test_user_ids)]

        bootstrap_df = self._bootstrap(precede_df, bootstrap_len)

        return (
            bootstrap_df
            .append(test_df)
            .sort_values(by=["user_id", "timestamp"])
            .reset_index(drop=True)
            .assign(item_idx=lambda df: df["item_id"].map(self.to_idx_map))
        )

    def recommend(self, user_id: int, k: int = 20):
        try:
            score = inference(user_id, self.bootstrapped_test_df, self.model, self.device)
            purchased = self.filter.filter_express(user_id)
            score[purchased] = -1.0
            _, indices = torch.topk(score, k)
            indices = indices.cpu().numpy()
            res = np.vectorize(self.to_id_map.get)(indices)
        except:
            res = self.cold_start.bestseller(size=k)
        return res.tolist()
