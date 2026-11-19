from pathlib import Path
from typing import Protocol

class ArtifactStore(Protocol):
    def save_model(self, model, filename) -> Path:
        ...

    def save_metadata(self, metadata, filename) -> Path:
        ...