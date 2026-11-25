from mini_mlops.factories.protocols import TrainableModel


class TrainingWorkflow:
    def run(self, model, X_train, y_train) -> TrainableModel:

        model.fit(X_train, y_train)

        return model  # ready Evaluation
