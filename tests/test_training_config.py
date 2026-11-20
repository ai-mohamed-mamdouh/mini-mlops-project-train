
import json
from pathlib import Path

import pytest

from mini_mlops.config.training_config import TrainingConfig
from mini_mlops.config.exceptions import ConfigurationError
from mini_mlops.config.loader import load_training_config


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def valid_config_data():
    """Return a valid training configuration dictionary."""

    return {
        "dataset_name": "wine",
        "test_size": 0.2,
        "random_state": 42,
        "model_name": "logistic_regression",
        "preprocessing": "standard_scaler",
        "metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ],
        "experiment_name": "test-experiment",
        "artifact_dir": "artifacts"
    }


@pytest.fixture
def valid_config_file(tmp_path, valid_config_data):
    """Create a temporary valid JSON configuration file."""

    path = tmp_path / "training_config.json"

    path.write_text(
        json.dumps(valid_config_data),
        encoding="utf-8"
    )

    return path


# ==========================================================
# 1. Valid Configuration
# ==========================================================

def test_valid_configuration(valid_config_file):

    config = load_training_config(valid_config_file)

    assert isinstance(config, TrainingConfig)

    assert config.dataset_name == "wine"
    assert config.test_size == 0.2
    assert config.random_state == 42
    assert config.model_name == "logistic_regression"
    assert config.preprocessing == "standard_scaler"

    assert config.metrics == [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
    assert config.experiment_name == "test-experiment"

    assert config.artifact_dir == str("artifacts")


# ==========================================================
# 2. Invalid Test Size
# ==========================================================

@pytest.mark.parametrize(
    "test_size",
    [0, 1, -0.2, 1.5]
)
def test_invalid_test_size(valid_config_data, test_size):

    config_data = valid_config_data.copy()
    config_data["test_size"] = test_size

    with pytest.raises(ConfigurationError):
        TrainingConfig.from_dict(config_data)


# ==========================================================
# 3. Negative Random State
# ==========================================================

def test_negative_random_state(valid_config_data):

    config_data = valid_config_data.copy()
    config_data["random_state"] = -1

    with pytest.raises(ConfigurationError):
        TrainingConfig.from_dict(config_data)


# ==========================================================
# 4. Empty Model Name
# ==========================================================

@pytest.mark.parametrize(
    "model_name",
    ["", "   "]
)
def test_empty_model_name(valid_config_data, model_name):

    config_data = valid_config_data.copy()
    config_data["model_name"] = model_name

    with pytest.raises(ConfigurationError):
        TrainingConfig.from_dict(config_data)


# ==========================================================
# 5. Empty Metrics
# ==========================================================

def test_empty_metrics(valid_config_data):

    config_data = valid_config_data.copy()
    config_data["metrics"] = []

    with pytest.raises(ConfigurationError):
        TrainingConfig.from_dict(config_data)


# ==========================================================
# 6. Duplicate Metrics
# ==========================================================

def test_duplicate_metrics(valid_config_data):

    config_data = valid_config_data.copy()

    config_data["metrics"] = [
        "accuracy",
        "precision",
        "accuracy"
    ]

    with pytest.raises(ConfigurationError):
        TrainingConfig.from_dict(config_data)


# ==========================================================
# 7. Missing JSON File
# ==========================================================

def test_missing_json_file(tmp_path):

    missing_file = tmp_path / "missing.json"

    with pytest.raises(ConfigurationError):
        load_training_config(missing_file)


# ==========================================================
# 8. Invalid JSON Syntax
# ==========================================================

def test_invalid_json_syntax(tmp_path):

    invalid_file = tmp_path / "invalid.json"

    invalid_file.write_text(
        '{"dataset_name": "wine",',
        encoding="utf-8"
    )

    with pytest.raises(ConfigurationError):
        load_training_config(invalid_file)


# ==========================================================
# 9. Missing Required Field
# ==========================================================

def test_missing_required_field(tmp_path, valid_config_data):

    config_data = valid_config_data.copy()

    del config_data["model_name"]

    config_file = tmp_path / "missing_field.json"

    config_file.write_text(
        json.dumps(config_data),
        encoding="utf-8"
    )

    with pytest.raises(ConfigurationError):
        load_training_config(config_file)
