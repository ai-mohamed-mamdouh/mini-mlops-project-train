from dataclasses import asdict
from functools import wraps

from mini_mlops.adapters.protocols import ArtifactStore, ExperimentTracker
from mini_mlops.config.training_config import TrainingConfig


def experiment_tracker(
    tracker: ExperimentTracker,
    run_name: str,
    config: TrainingConfig,
    *,
    artifact_store: ArtifactStore | None = None,
):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            tracker.start_run(run_name=run_name)

            status = "FAILED"

            try:
                model, result = func(*args, **kwargs)

                params = model.get_params()

                tracker.log_params(params)
                tracker.log_params(asdict(config))
                tracker.log_metrics(result)
                tracker.log_model(model, model_name=config.model_name)

                if artifact_store is not None:
                    artifact_store.save_model(
                        model=model,
                        filename=f"{run_name}.joblib",
                    )

                    artifact_store.save_metadata(
                        metadata={
                            "run_name": run_name,
                            "config": asdict(config),
                            "params": params,
                            "metrics": result,
                        },
                        filename=f"{run_name}.json",
                    )

                status = "FINISHED"

                return model, result

            finally:
                tracker.end_run(status=status)

        return wrapper

    return decorator
