from typing import Protocol, Any

class Metric(Protocol):

    name: str

    def calculate(
        self,
        y_true: Any,
        y_pred: Any
    ) -> float:
        ...