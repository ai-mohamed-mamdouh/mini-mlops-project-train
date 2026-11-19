from sklearn.base import BaseEstimator

class ModelFactory:

    def create(self, model_name:str) -> BaseEstimator :
        if model_name == 'LogisticRegression' :
            return LogisticRegression()
        
        elif model_name == 'RandomForestClassifier' :
            return RandomForestClassifier()
        else :
            raise ValueError('enter true model name ')