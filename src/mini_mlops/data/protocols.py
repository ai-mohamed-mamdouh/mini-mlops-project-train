from typing import Protocol

import pandas as pd


class DatasetLoader(Protocol):
    def load_data(self) -> tuple[pd.DataFrame, pd.Series]: ...
