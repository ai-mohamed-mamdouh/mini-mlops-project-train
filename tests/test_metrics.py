import pytest

from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.factories.metric_factory import MetricFactory


@pytest.mark.parametrize(
    "name, expected",
    [("accuracy", 0.75), ("precision", 5 / 6), ("recall", 0.75), ("f1", 11 / 15)],
)
def test_metric_values(name, expected):
    metric = MetricFactory().create(name)
    assert metric.name == name
    assert metric.calculate([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(expected)


def test_metric_factory_preserves_order():
    names = ["f1", "accuracy"]
    assert [m.name for m in MetricFactory().create_many(names)] == names


def test_unsupported_metric():
    with pytest.raises(ValueError, match="Unsupported metric"):
        MetricFactory().create("unknown")


def test_evaluator_returns_named_scores():
    metrics = MetricFactory().create_many(["accuracy", "f1"])
    assert ModelEvaluator().evaluate([0, 1], [0, 1], metrics) == {
        "accuracy": 1.0,
        "f1": 1.0,
    }
