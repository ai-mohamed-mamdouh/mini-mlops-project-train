from typing import Protocol

import numpy as np
import pandas as pd


class TrainableModel(Protocol):
    def fit(self, X_train: np.ndarray, y_train: pd.Series, **kwargs) -> None: ...
