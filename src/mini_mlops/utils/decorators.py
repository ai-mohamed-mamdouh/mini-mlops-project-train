from functools import wraps


def experiment_tracker(tracker, run_name, config):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            tracker.start_run(run_name=run_name)

            status = "FAILED"

            try:
                model, result = func(*args, **kwargs)

                tracker.log_params(model.get_params())
                tracker.log_params(config)

                tracker.log_metrics(result)

                tracker.log_model(model)

                status = "FINISHED"

                return model, result

            finally:
                tracker.end_run(status=status)

        return wrapper

    return decorator
