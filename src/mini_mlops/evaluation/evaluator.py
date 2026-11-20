
from mini_mlops.evaluation.protocols import Metric
import numpy as np
import pandas as pd
class ModelEvaluator:

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metrics: list[Metric]
    ) -> dict[str, float]:

        results = {}

        for metric in metrics:

            results[metric.name] = metric.calculate(
                y_true,
                y_pred
            )

        return results