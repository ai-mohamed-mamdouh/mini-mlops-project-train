from mini_mlops.data.loader import SklearnWineDatasetLoader


def test_dataset_loading():
    loader = SklearnWineDatasetLoader()

    X, _y = loader.load_data()

    assert X.shape[0] == 178


def test_dataset_features():
    loader = SklearnWineDatasetLoader()

    X, _y = loader.load_data()

    assert X.shape[1] == 13


def test_target_loading():
    loader = SklearnWineDatasetLoader()

    _X, y = loader.load_data()

    assert len(y) == 178


def test_local_loader_separates_features_and_target(tmp_path):
    import pandas as pd

    from mini_mlops.data.loader import LocalDatasetLoader

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    X, y = LocalDatasetLoader(path).load_data()
    pd.testing.assert_frame_equal(X, df.drop(columns="target"))
    pd.testing.assert_series_equal(y, df["target"])


def test_local_loader_missing_target(tmp_path):
    import pytest

    from mini_mlops.data.loader import LocalDatasetLoader

    path = tmp_path / "dataset.csv"
    path.write_text("feature\n1\n")
    with pytest.raises(KeyError):
        LocalDatasetLoader(path).load_data()


def test_local_loader_missing_file(tmp_path):
    import pytest

    from mini_mlops.data.loader import LocalDatasetLoader

    with pytest.raises(FileNotFoundError):
        LocalDatasetLoader(tmp_path / "missing.csv").load_data()
