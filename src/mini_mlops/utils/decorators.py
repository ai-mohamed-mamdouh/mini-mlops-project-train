from functools import wraps

def training_tracker(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Before function
        print("Before function")

        result = func(*args, **kwargs)

        # After function
        print("After function")

        return result

    return wrapper