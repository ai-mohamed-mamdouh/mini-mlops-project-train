from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

class StandardScalerStrategy:

    def create(self) -> BaseEstimator:
        return StandardScaler()