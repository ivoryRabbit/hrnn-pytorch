class Filter(object):
    def __init__(self, precede_df, item_df):
        self.precede_df = precede_df
        self.item_df = item_df

    def filter_express(self, user_id):
        return (
            self.precede_df
            .query(f"user_id=={user_id}")
            .filter("item_idx")
            .index
        )
