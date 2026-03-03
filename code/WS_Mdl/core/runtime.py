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
