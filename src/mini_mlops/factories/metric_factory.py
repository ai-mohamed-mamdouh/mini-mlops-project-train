
from mini_mlops.evaluation.protocols import Metric
from mini_mlops.evaluation.metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
)


class MetricFactory:

    def __init__(self):

        self._metrics = {
            "accuracy": AccuracyMetric,
            "precision": PrecisionMetric,
            "recall": RecallMetric,
            "f1": F1Metric,
        }

    def create(self, metric_name: str) -> Metric:

        metric_class = self._metrics.get(metric_name)

        if metric_class is None:
            raise ValueError(
                f"Unsupported metric: {metric_name}"
            )

        return metric_class()

    def create_many(self, metric_names: list[str]) -> list[Metric]:

        return [
            self.create(metric_name)
            for metric_name in metric_names
        ]
