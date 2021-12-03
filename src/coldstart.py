class Bestseller(object):
    def __init__(self, precede_df):
        self.precede_df = precede_df
        self.bestseller = self._get_bestseller()

    def _get_bestseller(self):
        pop = (
            self.precede_df
            .drop_duplicates(subset=["user_id", "item_id"])
            .groupby("item_id", as_index=False)["user_id"]
            .agg({"pop": "nunique"})
            .sort_values("pop", ascending=False)
        )
        return pop["item_id"].values

    def inference(self, size: int):
        return self.bestseller[:size]
