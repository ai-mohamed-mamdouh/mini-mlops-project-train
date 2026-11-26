from unittest.mock import Mock, call, sentinel

import pytest

from mini_mlops.workflows.experiment_pipeline import ExperimentPipeline


@pytest.fixture
def pipeline_setup():
    calls = Mock()
    calls.loader.load_data.return_value = (sentinel.X, sentinel.y)
    calls.splitter.split.return_value = (
        sentinel.X_train,
        sentinel.X_test,
        sentinel.y_train,
        sentinel.y_test,
    )
    calls.strategy.create.return_value = calls.processor
    calls.processor.fit_transform.return_value = sentinel.scaled_train
    calls.processor.transform.return_value = sentinel.scaled_test
    calls.training.run.return_value = sentinel.trained_model
    calls.evaluation.run.return_value = {"accuracy": 0.9}
    pipeline = ExperimentPipeline(
        calls.loader,
        calls.splitter,
        calls.strategy,
        sentinel.model,
        calls.training,
        calls.evaluation,
    )
    return pipeline, calls


def test_pipeline_order_and_data_flow(pipeline_setup):
    pipeline, calls = pipeline_setup
    assert pipeline.run() == (sentinel.trained_model, {"accuracy": 0.9})
    assert calls.mock_calls == [
        call.loader.load_data(),
        call.splitter.split(sentinel.X, sentinel.y),
        call.strategy.create(),
        call.processor.fit_transform(sentinel.X_train),
        call.processor.transform(sentinel.X_test),
        call.training.run(sentinel.model, sentinel.scaled_train, sentinel.y_train),
        call.evaluation.run(
            sentinel.trained_model, sentinel.scaled_test, sentinel.y_test
        ),
    ]


@pytest.mark.parametrize("stage", ["load", "split", "preprocess", "train"])
def test_pipeline_stops_before_evaluation_on_failure(pipeline_setup, stage):
    pipeline, calls = pipeline_setup
    failing_method = {
        "load": calls.loader.load_data,
        "split": calls.splitter.split,
        "preprocess": calls.processor.fit_transform,
        "train": calls.training.run,
    }[stage]
    failing_method.side_effect = RuntimeError("stage failed")
    with pytest.raises(RuntimeError, match="stage failed"):
        pipeline.run()
    calls.evaluation.run.assert_not_called()
    if stage != "train":
        calls.training.run.assert_not_called()
