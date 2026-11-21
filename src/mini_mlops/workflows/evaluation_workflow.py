import numpy as np
import pandas as pd

from mini_mlops.config.training_config import TrainingConfig
from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.factories.metric_factory import MetricFactory
from mini_mlops.factories.protocols import TrainableModel


class EvaluationWorkFlow:
    def __init__(
        self,
        config: TrainingConfig,
        model: TrainableModel,
        evaluator: ModelEvaluator,
        metric_factory: MetricFactory,
        X_test: np.ndarray,
        y_test: pd.Series,
    ):
        self.config = config
        self.model = model
        self.evaluator = evaluator
        self.metric_factory = metric_factory
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        y_pred = self.model.predict(self.X_test)

        metrics = self.metric_factory.create_many(self.config.metrics)

        results = self.evaluator.evaluate(self.y_test, y_pred, metrics)
        return results
