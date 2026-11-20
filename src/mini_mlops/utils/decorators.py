
from functools import wraps
from dataclasses import asdict


def experiment_tracker(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):

        tracker = self.tracker

        tracker.start_run(
            run_name=self.run_name
        )

        status = "FAILED"

        try:
            # Log configuration
            tracker.log_params(asdict(self.config))

            # Execute pipeline
            result = func(self, *args, **kwargs)

            status = "FINISHED"

            return result

        finally:
            tracker.end_run(status=status)

    return wrapper
