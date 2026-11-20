import pandas as pd
import numpy as np
from dataclasses import asdict
from sklearn.base import BaseEstimator
from mini_mlops.data.protocols import DatasetLoader
from mini_mlops.data.splitter import TrainTestSplitter
from mini_mlops.utils.decorators import training_tracker
from mini_mlops.adapters.protocols import ArtifactStore
from mini_mlops.config.training_config import TrainingConfig
from mini_mlops.preprocessing.protocols import PreprocessingStrategy
class TrainingWorkflow:

    def __init__(self,
                 config:TrainingConfig ,
                 data_loader:DatasetLoader,
                 splitter:TrainTestSplitter,
                 model:BaseEstimator,
                 pre_processor:PreprocessingStrategy,
                 artifact_store:ArtifactStore ,
                 ):
        
        self.config = config
        self.data_loader = data_loader
        self.splitter = splitter
        self.pre_processor = pre_processor
        self.model = model
        self.artifact_store = artifact_store

    @training_tracker
    def run(self) -> tuple[BaseEstimator, np.ndarray,   pd.Series]:

        X,y = self.data_loader.load_data()
        X_train, X_test, y_train, y_test = self.splitter.split(X, y)

        processor = self.pre_processor.create()

        X_train = processor.fit_transform(X_train)
        X_test = processor.transform(X_test)

        self.model.fit(X_train, y_train)

        self.artifact_store.save_model(
            self.model,
            "model.joblib"
        )
        self.artifact_store.save_metadata(
            asdict(self.config) ,
            "model_metadata.json" )

        return self.model , X_test, y_test # ready Evaluation
    