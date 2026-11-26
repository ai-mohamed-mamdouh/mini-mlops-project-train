import pandas as pd

from mini_mlops.data.splitter import TrainTestSplitter


def test_split_is_disjoint_aligned_stratified_and_reproducible():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0, 1] * 50)
    splitter = TrainTestSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(X.index)
    pd.testing.assert_series_equal(y_train, y.loc[X_train.index])
    pd.testing.assert_series_equal(y_test, y.loc[X_test.index])
    assert y_train.value_counts().to_dict() == {0: 40, 1: 40}
    assert y_test.value_counts().to_dict() == {0: 10, 1: 10}
    for first, second in zip((X_train, X_test, y_train, y_test), splitter.split(X, y)):
        assert first.equals(second)
