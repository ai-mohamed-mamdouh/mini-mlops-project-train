from configs.load_training_config import load_training_config
from pathlib import Path

config = load_training_config(Path("configs/training.json"))
print(config.model_name)