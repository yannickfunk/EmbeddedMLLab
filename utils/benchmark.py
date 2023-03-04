from functools import wraps
from timeit import timeit

def measure_execution(original_function=None, *, repetitions=75):
    def _decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed_time = timeit(lambda: func(*args, **kwargs), number=repetitions)
            print("Elapsed time: {:.6f} seconds".format(elapsed_time))
            return elapsed_time/repetitions
        return wrapper
    if original_function:
        return _decorate(original_function)
    return _decorate