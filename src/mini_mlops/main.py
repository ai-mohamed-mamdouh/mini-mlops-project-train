import os
from pathlib import Path

from dotenv import load_dotenv

from mini_mlops.adapters.artifact_stores import LocalArtifactStore
from mini_mlops.adapters.experiment_trackers import MLflowTracker
from mini_mlops.config.loader import load_training_config
from mini_mlops.data.loader import SklearnWineDatasetLoader
from mini_mlops.data.splitter import TrainTestSplitter
from mini_mlops.evaluation.evaluator import ModelEvaluator
from mini_mlops.factories.metric_factory import MetricFactory
from mini_mlops.factories.model_factory import ModelFactory
from mini_mlops.preprocessing.strategies import StandardScalerStrategy
from mini_mlops.workflows.experiment_pipeline import ExperimentPipeline
from mini_mlops.workflows.train_workflow import TrainingWorkflow


def main():

    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    # Configuration
    config = load_training_config(Path("configs/training.json"))

    # Dataset
    data_loader = SklearnWineDatasetLoader()

    # Splitter
    splitter = TrainTestSplitter(
        test_size=config.test_size, random_state=config.random_state
    )

    # Model
    model = ModelFactory().create(config.model_name, config.model_params)

    # Preprocessing
    pre_processor = StandardScalerStrategy()

    # Artifact Store
    artifact_store = LocalArtifactStore(Path(config.artifact_dir))

    # Training Workflow
    training_workflow = TrainingWorkflow(
        config=config,
        data_loader=data_loader,
        splitter=splitter,
        model=model,
        pre_processor=pre_processor,
        artifact_store=artifact_store,
    )

    # Evaluation Dependencies
    evaluator = ModelEvaluator()
    metric_factory = MetricFactory()

    # MLflow Tracker
    tracker = MLflowTracker(
        tracking_uri=tracking_uri, experiment_name=config.experiment_name
    )

    # Experiment Pipeline
    pipeline = ExperimentPipeline(
        config=config,
        training_workflow=training_workflow,
        evaluator=evaluator,
        metric_factory=metric_factory,
        tracker=tracker,
    )

    results = pipeline.run()

    print("Evaluation Results: ")
    print(results)


if __name__ == "__main__":
    main()
