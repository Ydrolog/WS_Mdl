import importlib
import time

from WS_Mdl.core.style import sprint


def timed_import(module):
    """e.g. from WS_Mdl import core as C; C.timed_import('imod')"""
    if not isinstance(module, str):
        module = module.__name__

    start = time.perf_counter()
    mod = importlib.import_module(module)
    end = time.perf_counter()

    sprint(f'{module} imported in {end - start:.3f}s')
    return mod


def timed_Exe(func, *args, pre=None, post='', verbose_in=False, verbose_out=False, **kwargs):
    """
    e.g. from WS_Mdl import core as C; C.timed_Exe(C.get_Mdl, MdlN)
    Trick: use pre='' if you only want to get the time in brackets.
    """
    if pre is None:
        sprint(f'{func.__name__} executed in: ', verbose_in=verbose_in, verbose_out=verbose_out, end='')
    else:
        sprint(f'{pre} ', verbose_in=verbose_in, verbose_out=verbose_out, end='')

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    sprint(f'{post} [{end - start:.1f}s]', verbose_in=verbose_in, verbose_out=verbose_out)

    return result


def timed_class_init(cls, *args, **kwargs):
    """e.g. from WS_Mdl import core as C; C.timed_class_init(C.Mdl_N, MdlN)"""
    start = time.perf_counter()
    instance = cls(*args, **kwargs)
    end = time.perf_counter()
    sprint(f'{cls.__name__} instance created in {end - start:.2f}s')
    return instance
