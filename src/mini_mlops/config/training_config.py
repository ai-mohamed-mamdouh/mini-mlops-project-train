from dataclasses import dataclass, field
from typing import Any

from .exceptions import ConfigurationError


@dataclass
class TrainingConfig:
    dataset_name: str
    test_size: float
    random_state: int
    model_name: str
    preprocessing: str
    metrics: list[str]
    artifact_dir: str
    experiment_name: str
    model_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if (
            isinstance(self.test_size, bool)
            or not isinstance(self.test_size, (int, float))
            or not 0 < self.test_size < 1
        ):
            raise ConfigurationError("test_size must be between 0 and 1.")

        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ConfigurationError("random_state must be a non-negative integer.")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ConfigurationError("model_name cannot be empty.")

        if not isinstance(self.metrics, list) or not self.metrics:
            raise ConfigurationError("metrics cannot be empty.")

        if any(
            not isinstance(metric, str) or not metric.strip() for metric in self.metrics
        ):
            raise ConfigurationError("metrics must contain non-empty strings.")

        if len(self.metrics) != len(set(self.metrics)):
            raise ConfigurationError("Duplicate metrics are not allowed.")

    @classmethod
    def from_dict(cls, config_data: dict[str, Any]) -> "TrainingConfig":

        if not isinstance(config_data, dict):
            raise ConfigurationError("Configuration must be a dictionary.")

        data = config_data.copy()

        try:
            if "artifact_dir" in data:
                data["artifact_dir"] = data["artifact_dir"]

            return cls(**data)

        except ConfigurationError:
            raise

        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid training configuration: {exc}") from exc
