import json
from pathlib import Path
from configs.training_config import TrainingConfig
from configs.exceptions import ConfigurationError

def load_training_config(path: Path | str) -> TrainingConfig:
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Configuration file not found: {path}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON format in: {path}"
        ) from exc

    except OSError as exc:
        raise ConfigurationError(
            f"Could not read configuration file: {exc}"
        ) from exc

    if not isinstance(config_data, dict):
        raise ConfigurationError(
            "Configuration file must contain a JSON object."
        )

    try:
        return TrainingConfig.from_dict(config_data)

    except (TypeError, ValueError, KeyError) as exc:
        raise ConfigurationError(
            f"Invalid training configuration: {exc}"
        ) from exc