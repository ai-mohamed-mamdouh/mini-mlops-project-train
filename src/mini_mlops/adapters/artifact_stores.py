import json
from pathlib import Path
from typing import Any

import joblib


class LocalArtifactStore:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, filename: str) -> Path:

        file_path = self.artifact_dir / filename
        joblib.dump(model, file_path)

        return file_path

    def save_metadata(self, metadata: dict[str, Any], filename: str) -> Path:

        file_path = self.artifact_dir / filename
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        return file_path
