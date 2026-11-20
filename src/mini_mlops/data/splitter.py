from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class TrainTestSplitter:

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    )->list[ pd.DataFrame | pd.Series]:
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )