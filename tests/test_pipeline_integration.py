import json
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from mini_mlops.adapters.artifact_stores import LocalArtifactStore
from mini_mlops.data.loader import LocalDatasetLoader, SklearnWineDatasetLoader
from mini_mlops.data.splitter import TrainTestSplitter
from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.factories.metric_factory import MetricFactory
from mini_mlops.factories.model_factory import ModelFactory
from mini_mlops.preprocessing.strategies import StandardScalerStrategy
from mini_mlops.utils.decorators import experiment_tracker
from mini_mlops.workflows.evaluation_workflow import EvaluationWorkFlow
from mini_mlops.workflows.experiment_pipeline import ExperimentPipeline
from mini_mlops.workflows.train_workflow import TrainingWorkflow


@pytest.mark.parametrize(
    "model_name,params",
    [
        ("LogisticRegression", {"max_iter": 1000}),
        ("RandomForestClassifier", {"n_estimators": 10, "random_state": 42}),
    ],
)
def test_real_pipeline_and_artifact_roundtrip(
    tmp_path, training_config, model_name, params
):
    X, y = SklearnWineDatasetLoader().load_data()
    path = tmp_path / "wine.csv"
    pd.concat([X, y.rename("target")], axis=1).to_csv(path, index=False)
    training_config.model_name = model_name
    training_config.model_params = params
    model = ModelFactory().create(model_name, params)
    pipeline = ExperimentPipeline(
        LocalDatasetLoader(path),
        TrainTestSplitter(0.2, 42),
        StandardScalerStrategy(),
        model,
        TrainingWorkflow(),
        EvaluationWorkFlow(
            training_config,
            ModelEvaluator(),
            MetricFactory().create_many(training_config.metrics),
        ),
    )
    tracker = Mock()
    store = LocalArtifactStore(tmp_path / "artifacts")
    run = experiment_tracker(tracker, "wine", training_config, artifact_store=store)(
        pipeline.run
    )
    trained, results = run()
    assert trained is model
    check_is_fitted(trained)
    assert set(results) == set(training_config.metrics)
    assert all(0 <= value <= 1 for value in results.values())
    assert results["accuracy"] > 0.6
    restored = joblib.load(tmp_path / "artifacts/wine.joblib")
    check_is_fitted(restored)
    assert restored.get_params() == trained.get_params()
    metadata = json.loads((tmp_path / "artifacts/wine.json").read_text())
    assert metadata["metrics"] == results
    tracker.end_run.assert_called_once_with(status="FINISHED")
