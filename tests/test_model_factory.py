import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from mini_mlops.factories.model_factory import ModelFactory


def test_logistic_regression_creation():

    factory = ModelFactory()

    model = factory.create(
        "LogisticRegression",
        {"max_iter": 1000}
    )

    assert isinstance(model, LogisticRegression)


def test_random_forest_creation():

    factory = ModelFactory()

    model = factory.create(
        "RandomForestClassifier",
        {"n_estimators": 100, "random_state": 42}
    )

    assert isinstance(model, RandomForestClassifier)


def test_unsupported_model():

    factory = ModelFactory()

    with pytest.raises(ValueError):
        factory.create("unsupported_model", {})