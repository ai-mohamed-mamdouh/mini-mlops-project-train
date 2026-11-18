import json
from pathlib import Path
from configs.training_config import TrainingConfig

def load_training_config(path: Path | str) -> TrainingConfig :

    with open(path, encoding="utf-8") as file:
        config_data = json.load(file)

    config = TrainingConfig.from_dict(config_data)
    return config