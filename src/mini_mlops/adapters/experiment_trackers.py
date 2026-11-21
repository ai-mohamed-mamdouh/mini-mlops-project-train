import json
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.sklearn

from mini_mlops.factories.protocols import TrainableModel


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "mini-mlops",
    ):

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None) -> str:

        run = mlflow.start_run(run_name=run_name)

        return run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:

        serialized_params = self._serialize_params(params)

        mlflow.log_params(serialized_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:

        mlflow.log_metrics(
            {name: float(value) for name, value in metrics.items()}, step=step
        )

    def log_model(
        self,
        model: TrainableModel,
        model_name: str = "model",
    ) -> None:

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    def log_artifact(self, local_path: Path, artifact_path: str | None = None) -> None:

        if not local_path.is_file():
            raise FileNotFoundError(local_path)

        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def end_run(
        self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED"
    ) -> None:

        mlflow.end_run(status=status)

    @staticmethod
    def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:

        serialized = {}

        for key, value in params.items():
            if isinstance(value, Path):
                value = str(value)

            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True, default=str)

            elif value is None:
                value = "None"

            serialized[key] = value

        return serialized
