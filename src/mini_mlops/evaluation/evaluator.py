from collections.abc import Sequence

from numpy.typing import ArrayLike

from mini_mlops.evaluation.protocols import Metric


class ModelEvaluator:
    def evaluate(
        self, y_true: ArrayLike, y_pred: ArrayLike, metrics: Sequence[Metric]
    ) -> dict[str, float]:

        results = {}

        for metric in metrics:
            results[metric.name] = metric.calculate(y_true, y_pred)

        return results
