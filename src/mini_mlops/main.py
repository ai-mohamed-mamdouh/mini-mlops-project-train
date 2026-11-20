from pathlib import Path

from mini_mlops.config.loader import load_training_config
from mini_mlops.data.splitter import TrainTestSplitter
from mini_mlops.data.loader import SklearnWineDatasetLoader
from mini_mlops.workflows.train_workflow import TrainingWorkflow
from mini_mlops.adapters.artifact_stores import LocalArtifactStore
from mini_mlops.preprocessing.strategies import StandardScalerStrategy
from mini_mlops.factories.model_factory import ModelFactory


# 1. Configuration
config = load_training_config(Path("configs/training.json"))

# 2. Dataset Loader
data_loader = SklearnWineDatasetLoader()

# 3. Data Splitter
splitter = TrainTestSplitter(
    test_size=config.test_size,
    random_state=config.random_state
)

# 4. Model
model = ModelFactory().create(
    config.model_name,
    config.model_params
)

# 5. Preprocessing Strategy
preprocessor_strategy = StandardScalerStrategy()

# 6. Artifact Store
artifact_store = LocalArtifactStore(
    Path("artifacts")
)

# 7. Training Workflow
workflow = TrainingWorkflow(
    config=config,
    data_loader=data_loader,
    splitter=splitter,
    model=model,
    pre_processor=preprocessor_strategy,
    artifact_store=artifact_store
)

# 8. Run
model , X_test, y_test = workflow.run()
print(type(X_test), type(y_test))
print(type(model))
