import os
from pathlib import Path

from dotenv import load_dotenv

from mini_mlops.adapters.artifact_stores import LocalArtifactStore
from mini_mlops.adapters.experiment_trackers import MLflowTracker
from mini_mlops.config.loader import load_training_config
from mini_mlops.data.loader import LocalDatasetLoader
from mini_mlops.data.splitter import TrainTestSplitter
from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.factories.metric_factory import MetricFactory
from mini_mlops.factories.model_factory import ModelFactory
from mini_mlops.preprocessing.strategies import StandardScalerStrategy
from mini_mlops.utils.decorators import experiment_tracker
from mini_mlops.workflows.evaluation_workflow import EvaluationWorkFlow
from mini_mlops.workflows.experiment_pipeline import ExperimentPipeline
from mini_mlops.workflows.train_workflow import TrainingWorkflow


def main() -> None:
    load_dotenv()

    # Configuration
    config_path = Path(os.getenv("TRAINING_CONFIG_PATH", "configs/training.json"))
    DATA_PATH = Path(os.environ["DATA_PATH"])
    # ARTIFACT_DIR = str(os.environ["ARTIFACT_DIR"])
    MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
    config = load_training_config(config_path)

    # Data Dependencies
    data_loader = LocalDatasetLoader(data_path=DATA_PATH)

    splitter = TrainTestSplitter(
        test_size=config.test_size,
        random_state=config.random_state,
    )

    pre_processor = StandardScalerStrategy()

    # Model
    model = ModelFactory().create(
        config.model_name,
        config.model_params,
    )

    # Evaluation Dependencies
    evaluator = ModelEvaluator()
    metrics = MetricFactory().create_many(config.metrics)

    # Workflows
    training_workflow = TrainingWorkflow()

    evaluation_workflow = EvaluationWorkFlow(
        config=config,
        evaluator=evaluator,
        metrics=metrics,
    )

    # Experiment Pipeline
    experiment_pipeline = ExperimentPipeline(
        data_loader=data_loader,
        splitter=splitter,
        pre_processor=pre_processor,
        model=model,
        training_workflow=training_workflow,
        evaluation_workflow=evaluation_workflow,
    )

    # Optional Experiment Tracking
    run = experiment_pipeline.run

    if config.enable_tracking:
        tracker = MLflowTracker(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=config.experiment_name,
        )

        run = experiment_tracker(
            tracker=tracker,
            run_name="wine_experiment",
            config=config,
            artifact_store=LocalArtifactStore(
                artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts"))
            ),
        )(run)

    # Execute Experiment
    model, result = run()

    print("Evaluation Results:")
    print(result)


if __name__ == "__main__":
    main()
