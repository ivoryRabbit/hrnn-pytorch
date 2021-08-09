class ColdStart(object):
    def __init__(self, precede_df):
        self.precede_df = precede_df

    def bestseller(self, size: int = 10):
        pop = (
            self.precede_df
            .drop_duplicates(subset=["user_id", "item_id"])
            .groupby("item_id", as_index=False)["user_id"]
            .agg({"pop": "nunique"})
            .sort_values("pop", ascending=False)
        )
        return pop["item_id"].values[:size]