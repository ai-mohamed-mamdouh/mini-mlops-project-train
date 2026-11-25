from mini_mlops.data.protocols import DatasetLoader
from mini_mlops.data.splitter import TrainTestSplitter
from mini_mlops.factories.protocols import TrainableModel
from mini_mlops.preprocessing.protocols import PreprocessingStrategy


class ExperimentPipeline:
    def __init__(
        self,
        data_loader: DatasetLoader,
        splitter: TrainTestSplitter,
        pre_processor: PreprocessingStrategy,
        model: TrainableModel,
        training_workflow,
        evaluation_workflow,
    ):

        self.data_loader = data_loader
        self.splitter = splitter
        self.pre_processor = pre_processor
        self.model = model
        self.training_workflow = training_workflow
        self.evaluation_workflow = evaluation_workflow

    def run(self):

        X, y = self.data_loader.load_data()
        X_train, X_test, y_train, y_test = self.splitter.split(X, y)

        processor = self.pre_processor.create()

        X_train = processor.fit_transform(X_train)
        X_test = processor.transform(X_test)

        # Training
        model = self.training_workflow.run(self.model, X_train, y_train)

        # Evaluation
        results = self.evaluation_workflow.run(model, X_test, y_test)

        return model, results
