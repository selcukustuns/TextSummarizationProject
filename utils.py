def save_data(data, path):
    data.to_parquet(path, index=False)

def load_data(path):
    return pd.read_parquet(path)
