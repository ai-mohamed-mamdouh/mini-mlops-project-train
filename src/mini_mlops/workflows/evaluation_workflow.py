from collections.abc import Sequence

from mini_mlops.config.training_config import TrainingConfig
from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.evaluation.protocols import Metric


class EvaluationWorkFlow:
    def __init__(
        self,
        config: TrainingConfig,
        evaluator: ModelEvaluator,
        metrics: Sequence[Metric],
    ):
        self.config = config
        self.evaluator = evaluator
        self.metrics = metrics

    def run(self, model, X_test, y_test) -> dict[str, float]:
        y_pred = model.predict(X_test)

        results = self.evaluator.evaluate(y_test, y_pred, self.metrics)
        return results
