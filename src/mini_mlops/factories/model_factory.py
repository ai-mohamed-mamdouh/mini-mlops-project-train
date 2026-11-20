from .protocols import TrainableModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Any
class ModelFactory:

    def create(self, model_name:str, model_params : dict[str, Any]) -> TrainableModel :
        if model_name == 'LogisticRegression' :
            return LogisticRegression(**model_params)
        
        elif model_name == 'RandomForestClassifier' :
            return RandomForestClassifier(**model_params)
        else :
            raise ValueError('enter true model name ')