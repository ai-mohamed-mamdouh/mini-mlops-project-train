from pathlib import Path
from typing import Protocol
from sklearn.base import BaseEstimator
from typing import Literal, Any
from mini_mlops.factories.protocols import TrainableModel
class ArtifactStore(Protocol):
    def save_model(self, model, filename) -> Path:
        ...

    def save_metadata(self, metadata, filename) -> Path:
        ...


class ExperimentTracker(Protocol):

    def start_run(self, run_name: str | None = None) -> str:
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        ...

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None
    ) -> None:
        ...

    def log_model(
        self,
        model: TrainableModel,
        model_name: str = "model"
    ) -> None:
        ...

    def log_artifact(
        self,
        local_path: Path,
        artifact_path: str | None = None
    ) -> None:
        ...

    def end_run(
        self,
        status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED"
    ) -> None:
        ...
