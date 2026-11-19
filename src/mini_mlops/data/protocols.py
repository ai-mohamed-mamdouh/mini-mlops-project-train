import pandas as pd
from pathlib import Path
from typing import Protocol, Any

class DatasetLoader(Protocol):
    def load_data( self ) -> tuple[pd.DataFrame, pd.Series]:
        ...