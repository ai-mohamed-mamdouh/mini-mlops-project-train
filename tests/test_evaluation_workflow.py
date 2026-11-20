
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from unittest.mock import Mock

from mini_mlops.workflows.evaluation_workflow import EvaluationWorkFlow
from mini_mlops.config.training_config import TrainingConfig


@pytest.fixture
def workflow_setup():

    config = TrainingConfig(
        dataset_name="wine",
        test_size=0.2,
        random_state=42,
        model_name="logistic_regression",
        preprocessing="standard_scaler",
        metrics=["accuracy", "precision", "recall", "f1"],
        experiment_name="test-experiment",
        artifact_dir=Path("artifacts")
    )

    X_test = np.array([
        [1.2, 2.3, 3.4],
        [4.5, 5.6, 6.7]
    ])

    y_test = pd.Series([0, 1])
    y_pred = np.array([0, 1])

    model = Mock()
    model.predict.return_value = y_pred

    metric_factory = Mock()

    metrics = [
        Mock(),
        Mock(),
        Mock(),
        Mock()
    ]

    metric_factory.create_many.return_value = metrics

    evaluator = Mock()

    expected_results = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
    }

    evaluator.evaluate.return_value = expected_results

    workflow = EvaluationWorkFlow(
        config=config,
        model=model,
        evaluator=evaluator,
        metric_factory=metric_factory,
        X_test=X_test,
        y_test=y_test
    )

    return {
        "workflow": workflow,
        "config": config,
        "model": model,
        "evaluator": evaluator,
        "metric_factory": metric_factory,
        "metrics": metrics,
        "experiment_name": config.experiment_name,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "expected_results": expected_results
    }


def test_evaluation_workflow_initialization(workflow_setup):

    workflow = workflow_setup["workflow"]

    assert isinstance(workflow, EvaluationWorkFlow)


def test_model_injection(workflow_setup):

    workflow = workflow_setup["workflow"]

    assert workflow.model is workflow_setup["model"]


def test_evaluator_injection(workflow_setup):

    workflow = workflow_setup["workflow"]

    assert workflow.evaluator is workflow_setup["evaluator"]


def test_metric_factory_injection(workflow_setup):

    workflow = workflow_setup["workflow"]

    assert workflow.metric_factory is workflow_setup["metric_factory"]


def test_model_prediction(workflow_setup):

    workflow = workflow_setup["workflow"]
    model = workflow_setup["model"]

    workflow.run()

    model.predict.assert_called_once_with(
        workflow_setup["X_test"]
    )


def test_metric_factory_creation(workflow_setup):

    workflow = workflow_setup["workflow"]
    metric_factory = workflow_setup["metric_factory"]

    workflow.run()

    metric_factory.create_many.assert_called_once_with(
        workflow_setup["config"].metrics
    )


def test_evaluator_receives_correct_data(workflow_setup):

    workflow = workflow_setup["workflow"]
    evaluator = workflow_setup["evaluator"]

    workflow.run()

    evaluator.evaluate.assert_called_once_with(
        workflow_setup["y_test"],
        workflow_setup["y_pred"],
        workflow_setup["metrics"]
    )


def test_evaluation_returns_results(workflow_setup):

    workflow = workflow_setup["workflow"]

    results = workflow.run()

    assert results == workflow_setup["expected_results"]
