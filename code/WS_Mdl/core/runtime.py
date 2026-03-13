import importlib
import time


def timed_import(module):
    """e.g. from WS_Mdl import core as C; C.timed_import('imod')"""
    if not isinstance(module, str):
        module = module.__name__

    start = time.perf_counter()
    mod = importlib.import_module(module)
    end = time.perf_counter()

    print(f'{module} imported in {end - start:.3f}s')
    return mod


def timed_execution(func, *args, **kwargs):
    """e.g. from WS_Mdl import core as C; C.timed_execution(C.get_Mdl, MdlN)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f'{func.__name__} executed in {end - start:.3f}s')
    return result


def timed_class_init(cls, *args, **kwargs):
    """e.g. from WS_Mdl import core as C; C.timed_class_init(C.Mdl_N, MdlN)"""
    start = time.perf_counter()
    instance = cls(*args, **kwargs)
    end = time.perf_counter()
    print(f'{cls.__name__} instance created in {end - start:.3f}s')
    return instance
