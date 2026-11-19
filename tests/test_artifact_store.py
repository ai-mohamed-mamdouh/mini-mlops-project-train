import json

from pathlib import Path
from sklearn.linear_model import LogisticRegression

from mini_mlops.adapters.artifact_stores import (
    LocalArtifactStore
)


def test_save_model(tmp_path):

    store = LocalArtifactStore(tmp_path)

    model = LogisticRegression()

    saved_path = store.save_model(
        model,
        "wine_model.joblib"
    )

    assert saved_path.exists()


def test_save_metadata(tmp_path):

    store = LocalArtifactStore(tmp_path)

    metadata = {
        "model_name": "LogisticRegression",
        "dataset": "Wine",
        "accuracy": 0.95
    }

    saved_path = store.save_metadata(
        metadata,
        "metadata.json"
    )

    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)

    assert loaded_metadata == metadata