from mini_mlops.workflows.evaluation_workflow import EvaluationWorkFlow
from mini_mlops.utils.decorators import experiment_tracker


class ExperimentPipeline:

    def __init__(
        self,
        config,
        training_workflow,
        evaluator,
        metric_factory,
        tracker
    ):
        self.config = config
        self.training_workflow = training_workflow
        self.evaluator = evaluator
        self.metric_factory = metric_factory
        self.tracker = tracker
        self.run_name = config.model_name

    @experiment_tracker
    def run(self):

        # Training
        model, X_test, y_test = self.training_workflow.run()

        # Evaluation
        evaluation_workflow = EvaluationWorkFlow(
            config=self.config,
            model=model,
            evaluator=self.evaluator,
            metric_factory=self.metric_factory,
            X_test=X_test,
            y_test=y_test
        )

        results = evaluation_workflow.run()

        # Log Model
        self.tracker.log_model(
            model,
            model_name=self.config.model_name
        )

        # Log Metrics
        self.tracker.log_metrics(results)

        return results