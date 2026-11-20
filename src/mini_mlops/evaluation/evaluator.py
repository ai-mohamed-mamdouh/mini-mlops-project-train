
from mini_mlops.evaluation.protocols import Metric

class ModelEvaluator:

    def evaluate(
        self,
        y_true,
        y_pred,
        metrics: list[Metric]
    ) -> dict[str, float]:

        results = {}

        for metric in metrics:

            results[metric.name] = metric.calculate(
                y_true,
                y_pred
            )

        return results