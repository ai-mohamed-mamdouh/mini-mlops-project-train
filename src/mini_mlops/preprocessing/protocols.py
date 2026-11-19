from typing import Protocol
from sklearn.base import BaseEstimator

class PreprocessingStratgy(Protocol):
    def create(self) -> BaseEstimator:
        ...
