import importlib
import time


def timed_import(module):

    if not isinstance(module, str):
        module = module.__name__

    start = time.perf_counter()
    mod = importlib.import_module(module)
    end = time.perf_counter()

    print(f'{module} imported in {end - start:.3f}s')
    return mod


def timed_execution(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f'{func.__name__} executed in {end - start:.3f}s')
    return result
