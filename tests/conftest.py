import pytest

from mini_mlops.config.training_config import TrainingConfig


@pytest.fixture
def training_config():
    return TrainingConfig(
        dataset_name="wine",
        test_size=0.2,
        random_state=42,
        model_name="LogisticRegression",
        preprocessing="standard_scaler",
        metrics=["accuracy", "precision", "recall", "f1"],
        artifact_dir="artifacts",
        experiment_name="test-experiment",
        model_params={"max_iter": 1000},
        enable_tracking=True,
    )
