def bestseller(df, user_key="user_id", item_key="item_id", eval_k=20):
    df = df[[user_key, item_key]].drop_duplicates()
    pop = (
        df
        .groupby(item_key, as_index=False)[user_key]
        .agg({"pop": "count"})
        .sort_values("pop", ascending=False)
    )
    return pop[item_key].values[:eval_k]