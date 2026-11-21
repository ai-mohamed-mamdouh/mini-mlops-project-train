import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mini_mlops.adapters.experiment_trackers import MLflowTracker


@pytest.fixture
def tracker():

    with (
        patch("mini_mlops.adapters.experiment_trackers.mlflow.set_tracking_uri"),
        patch("mini_mlops.adapters.experiment_trackers.mlflow.set_experiment"),
    ):
        return MLflowTracker(
            tracking_uri="http://127.0.0.1:5000", experiment_name="test-experiment"
        )


# 1. Initialization


def test_tracker_initialization():

    with (
        patch(
            "mini_mlops.adapters.experiment_trackers.mlflow.set_tracking_uri"
        ) as set_uri,
        patch(
            "mini_mlops.adapters.experiment_trackers.mlflow.set_experiment"
        ) as set_experiment,
    ):
        tracker = MLflowTracker(
            tracking_uri="http://127.0.0.1:5000", experiment_name="test-experiment"
        )

        set_uri.assert_called_once_with("http://127.0.0.1:5000")
        set_experiment.assert_called_once_with("test-experiment")


# 2. Start Run


def test_start_run(tracker):

    mock_run = Mock()
    mock_run.info.run_id = "test-run-123"

    with patch(
        "mini_mlops.adapters.experiment_trackers.mlflow.start_run",
        return_value=mock_run,
    ) as start_run:
        run_id = tracker.start_run("wine-experiment")

        assert run_id == "test-run-123"

        start_run.assert_called_once_with(run_name="wine-experiment")


# 3. Log Parameters


def test_log_params(tracker):

    params = {
        "model_name": "random_forest",
        "n_estimators": 100,
        "artifact_dir": Path("artifacts"),
        "model_params": {"max_depth": 5},
        "optional_param": None,
    }

    expected_params = {
        "model_name": "random_forest",
        "n_estimators": 100,
        "artifact_dir": "artifacts",
        "model_params": json.dumps({"max_depth": 5}, sort_keys=True),
        "optional_param": "None",
    }

    with patch(
        "mini_mlops.adapters.experiment_trackers.mlflow.log_params"
    ) as log_params:
        tracker.log_params(params)

        log_params.assert_called_once_with(expected_params)


# 4. Log Metrics


def test_log_metrics(tracker):

    metrics = {"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1": 0.94}

    with patch(
        "mini_mlops.adapters.experiment_trackers.mlflow.log_metrics"
    ) as log_metrics:
        tracker.log_metrics(metrics, step=1)

        log_metrics.assert_called_once_with(metrics, step=1)


# 5. Log Model


def test_log_model(tracker):

    model = Mock()

    with patch(
        "mini_mlops.adapters.experiment_trackers.mlflow.sklearn.log_model"
    ) as log_model:
        tracker.log_model(model, model_name="wine-classifier")

        log_model.assert_called_once_with(
            sk_model=model,
            name="wine-classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )


# 6. Log Artifact


def test_log_artifact(tracker, tmp_path):

    artifact = tmp_path / "metadata.json"
    artifact.write_text('{"model": "random_forest"}')

    with patch(
        "mini_mlops.adapters.experiment_trackers.mlflow.log_artifact"
    ) as log_artifact:
        tracker.log_artifact(artifact, artifact_path="metadata")

        log_artifact.assert_called_once_with(str(artifact), artifact_path="metadata")


# 7. Log Non-existent Artifact


def test_log_nonexistent_artifact(tracker, tmp_path):

    artifact = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        tracker.log_artifact(artifact)


# 8. End Run


@pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
def test_end_run(tracker, status):

    with patch("mini_mlops.adapters.experiment_trackers.mlflow.end_run") as end_run:
        tracker.end_run(status=status)

        end_run.assert_called_once_with(status=status)
