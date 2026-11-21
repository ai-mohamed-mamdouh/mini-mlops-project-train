import pandas as pd
from sklearn.datasets import load_wine


class SklearnWineDatasetLoader:
    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        dataset = load_wine(as_frame=True)
        X = dataset.data
        y = dataset.target
        return (X, y)
