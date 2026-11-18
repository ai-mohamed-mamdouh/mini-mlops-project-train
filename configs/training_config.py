
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    dataset_name: str
    test_size: float
    random_state: int
    model_name: str
    preprocessing: str
    metrics: list[str]
    artifact_dir: Path

    @classmethod
    def from_dict(cls, config_data: dict[str, Any]) -> "TrainingConfig":
        data = config_data.copy()
        data["artifact_dir"] = Path(data["artifact_dir"])

        return cls(**data)

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if not self.metrics:
            raise ValueError("At least one evaluation metric is required.")
