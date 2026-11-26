import pandas as pd
from sklearn.datasets import load_wine


class SklearnWineDatasetLoader:
    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        dataset = load_wine(as_frame=True)
        X = dataset.data
        y = dataset.target
        return (X, y)


class LocalDatasetLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=["target"])
        y = df["target"]
        return (X, y)
