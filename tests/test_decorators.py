from dataclasses import asdict
from unittest.mock import Mock, call

import pytest

from mini_mlops.utils.decorators import experiment_tracker


@pytest.fixture
def tracking_setup(training_config):
    tracker = Mock()
    store = Mock()
    model = Mock()
    model.get_params.return_value = {"C": 1.0}
    result = {"accuracy": 0.9}
    func = Mock(return_value=(model, result))
    return tracker, store, model, result, func, training_config


@pytest.mark.parametrize("with_store", [False, True])
def test_tracking_success(tracking_setup, with_store):
    tracker, store, model, result, func, config = tracking_setup
    run = experiment_tracker(
        tracker, "run", config, artifact_store=store if with_store else None
    )(func)
    assert run(1, flag=True) == (model, result)
    func.assert_called_once_with(1, flag=True)
    assert tracker.mock_calls == [
        call.start_run(run_name="run"),
        call.log_params({"C": 1.0}),
        call.log_params(asdict(config)),
        call.log_metrics(result),
        call.log_model(model, model_name=config.model_name),
        call.end_run(status="FINISHED"),
    ]
    if with_store:
        store.save_model.assert_called_once_with(model=model, filename="run.joblib")
        store.save_metadata.assert_called_once_with(
            metadata={
                "run_name": "run",
                "config": asdict(config),
                "params": {"C": 1.0},
                "metrics": result,
            },
            filename="run.json",
        )
    else:
        assert not store.mock_calls


@pytest.mark.parametrize("stage", ["run", "log", "save_model", "save_metadata"])
def test_tracking_marks_failed_and_propagates_error(tracking_setup, stage):
    tracker, store, _, _, func, config = tracking_setup
    failing = {
        "run": func,
        "log": tracker.log_metrics,
        "save_model": store.save_model,
        "save_metadata": store.save_metadata,
    }[stage]
    failing.side_effect = RuntimeError("experiment failed")
    run = experiment_tracker(tracker, "run", config, artifact_store=store)(func)
    with pytest.raises(RuntimeError, match="experiment failed"):
        run()
    tracker.end_run.assert_called_once_with(status="FAILED")
    if stage in ("run", "log"):
        store.save_model.assert_not_called()
    if stage != "save_metadata":
        store.save_metadata.assert_not_called()


def test_start_run_failure_does_not_end_unopened_run(tracking_setup):
    tracker, store, _, _, func, config = tracking_setup
    tracker.start_run.side_effect = RuntimeError("start failed")
    run = experiment_tracker(tracker, "run", config, artifact_store=store)(func)
    with pytest.raises(RuntimeError, match="start failed"):
        run()
    func.assert_not_called()
    tracker.end_run.assert_not_called()


def test_decorator_preserves_function_metadata(training_config):
    def experiment():
        """Experiment docstring."""

    wrapped = experiment_tracker(Mock(), "run", training_config)(experiment)
    assert wrapped.__name__ == experiment.__name__
    assert wrapped.__doc__ == experiment.__doc__
