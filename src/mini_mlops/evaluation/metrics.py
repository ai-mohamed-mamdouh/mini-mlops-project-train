from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class AccuracyMetric:
    name = "accuracy"

    def calculate(self, y_true, y_pred) -> float:
        return float(accuracy_score(y_true, y_pred))


class PrecisionMetric:
    name = "precision"

    def calculate(self, y_true, y_pred) -> float:
        return float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )


class RecallMetric:
    name = "recall"

    def calculate(self, y_true, y_pred) -> float:
        return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


class F1Metric:
    name = "f1"

    def calculate(self, y_true, y_pred) -> float:
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
