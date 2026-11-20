import numpy as np
import pandas as pd
from typing import Protocol

class TrainableModel(Protocol):
     def fit(
         self,
         X_train: np.ndarray,
         y_train: pd.Series,
         **kwargs
     ) -> None:
         ...