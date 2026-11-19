from mini_mlops.data.loader import (
    SklearnWineDatasetLoader
)


def test_dataset_loading():
    loader = SklearnWineDatasetLoader()

    X, y = loader.load_data()

    assert X.shape[0] == 178


def test_dataset_features():
    loader = SklearnWineDatasetLoader()

    X, y = loader.load_data()

    assert X.shape[1] == 13


def test_target_loading():
    loader = SklearnWineDatasetLoader()

    X, y = loader.load_data()

    assert len(y) == 178