from unittest.mock import Mock

import numpy as np
import pytest

from mini_mlops.workflows.evaluation_workflow import EvaluationWorkFlow


@pytest.fixture
def evaluation_setup(training_config):
    evaluator = Mock(spec=["evaluate"])
    metrics = [Mock(), Mock()]
    model = Mock(spec=["predict"])
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    predictions = np.array([0, 0])
    model.predict.return_value = predictions
    evaluator.evaluate.return_value = {"accuracy": 0.5}
    workflow = EvaluationWorkFlow(training_config, evaluator, metrics)
    return workflow, model, evaluator, metrics, X, y, predictions


def test_evaluation_predicts_and_passes_correct_inputs(evaluation_setup):
    workflow, model, evaluator, metrics, X, y, predictions = evaluation_setup
    result = workflow.run(model, X, y)
    model.predict.assert_called_once_with(X)
    evaluator.evaluate.assert_called_once_with(y, predictions, metrics)
    assert result == {"accuracy": 0.5}


def test_prediction_failure_does_not_evaluate(evaluation_setup):
    workflow, model, evaluator, _, X, y, _ = evaluation_setup
    model.predict.side_effect = RuntimeError("prediction failed")
    with pytest.raises(RuntimeError, match="prediction failed"):
        workflow.run(model, X, y)
    evaluator.evaluate.assert_not_called()


def test_evaluation_failure_propagates(evaluation_setup):
    workflow, model, evaluator, _, X, y, _ = evaluation_setup
    evaluator.evaluate.side_effect = ValueError("metric failed")
    with pytest.raises(ValueError, match="metric failed"):
        workflow.run(model, X, y)
