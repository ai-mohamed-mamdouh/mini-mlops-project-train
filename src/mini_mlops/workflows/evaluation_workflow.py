import numpy as np
import pandas as pd
from dataclasses import asdict
from mini_mlops.factories.metric_factory import MetricFactory
from mini_mlops.factories.protocols import TrainableModel
from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.config.training_config import TrainingConfig

class EvaluationWorkFlow:
    def __init__(self,
                 config: TrainingConfig,
                 model: TrainableModel,
                 evaluator: ModelEvaluator,
                 X_test: np.ndarray,
                 y_test: pd.Series
                 ):
        self.config = config
        self.model = model
        self.evaluator = evaluator
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        y_pred = self.model.predict(self.X_test)
        metrics = self.config.metrics

        metric_factory = MetricFactory()
        metrics = metric_factory.create_many(self.config.metrics)


        results = self.evaluator.evaluate(
            self.y_test,
            y_pred,
            metrics
        )
        return results