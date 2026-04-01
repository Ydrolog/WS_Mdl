# ---------- Printing Section ----------

import re
import time

from colored import attr, fg

from .path import Pa_WS

__all__ = [
    'Sep',
    'Sep_2',
    'style_reset',
    'bold',
    'dim',
    'warn',
    'CuCh',
    'set_verbose',
    'sprint',
    'sinput',
    'green',
    'blue',
]

style_reset = f'{attr("reset")}\033[0m'
bold = '\033[1m'
dim = '\033[2m'
warn = f'\033[1m{fg("indian_red_1c")}'
green = f'{bold}{fg("green")}'
blue = f'{bold}{fg("blue")}'
Sep_N = 100  # Number of characters in a separator line.
Sep = f'{fg(52)}{"-" * Sep_N}{attr("reset")}\n'  # Main separator - for script start and end.
Sep_2 = f'{dim}{"-" * Sep_N}{attr("reset")}\n'  # Secondary separator - for sections within a script.


CuCh = {  # Stands for Custom Characters.
    '-': '🔴',  # negative
    '0': '🟡',  # neutral
    '+': '🟢',  # positive
    '=': '⚪️',  # no action required
    'x': '⚫️',  # already done
}

VERBOSE = True  # Use set_verbose to change this to False and get no information printed to the console.
START_TIME = 0.0


def set_verbose(v: bool):
    """Sets the VERBOSE variable to True or False."""
    global VERBOSE
    VERBOSE = v


l_Mdl = [i.name for i in (Pa_WS / 'models').iterdir()]


def _highlight_MdlN(text: str, style_In: str) -> str:
    patterns_re = '|'.join(re.escape(p) for p in sorted(l_Mdl, key=len, reverse=True))
    return re.sub(rf'({patterns_re})(\d+)', lambda m: f'{blue}{m.group(0)}{style_reset}{style_In}', text)


def sprint(
    *args,
    indent: int = 0,
    style: str = '',
    verbose_in: bool = None,
    verbose_out: bool = None,
    set_time: bool = False,
    print_time: bool = False,
    **kwargs,
):
    """
    Special print function. Allows easy indentation (2 spaces per 1 indent level) and easy styling.
    Allows for setting VERBOSE prior and after printing, so set_verbose doesn't have to be used all the time separately.
    Can also track and print elapsed time using set_time and print_time.
    """
    global START_TIME

    if verbose_in is not None:
        globals()['set_verbose'](verbose_in)

    time_str = ''
    if print_time:
        elapsed = time.time() - START_TIME
        time_str = f' [{elapsed:.1f} s]'

    if set_time:
        START_TIME = time.time()

    if VERBOSE:
        args_fmt = tuple(_highlight_MdlN(str(arg), style) for arg in args)  # Highlights Mdl_N instances.
        prefix = f'{style}{"  " * indent}'

        kwargs.setdefault('flush', True)  # Flush the output immediately, so it appears in the console without delay.
        if args_fmt:
            args_out = list(args_fmt)
            args_out[0] = f'{prefix}{args_out[0]}'
            args_out[-1] = f'{args_out[-1]}{time_str}{style_reset}'
            print(*args_out, **kwargs)
        else:
            print(f'{prefix}{time_str}{style_reset}', **kwargs)

    if verbose_out is not None:
        globals()['set_verbose'](verbose_out)


def sinput(*args, indent: int = 0, style: str = '', sep: str = ' '):
    """
    Special input function. Allows easy indentation and easy styling.
    Always prompts, regardless of VERBOSE.
    """
    args_fmt = (_highlight_MdlN(str(arg), style) for arg in args)
    prompt = f'{style}{"  " * indent}{sep.join(str(arg) for arg in args_fmt)}{style_reset}'
    return input(prompt)
