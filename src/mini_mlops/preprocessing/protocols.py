from typing import Protocol

from sklearn.base import BaseEstimator


class PreprocessingStrategy(Protocol):
    def create(self) -> BaseEstimator: ...
