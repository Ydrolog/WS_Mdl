# ---------- Printing Section ----------

import re
import time

from colored import attr, fg

from WS_Mdl.core.defaults import Pa_WS

__all__ = [
    'Sep',
    'Sep_2',
    'style_reset',
    'bold',
    'dim',
    'warn',
    'CuCh',
    'get_verbose',
    'set_verbose',
    'sprint',
    'sinput',
    'green',
    'blue',
    'RED',
]

style_reset = f'{attr("reset")}\033[0m'
bold = '\033[1m'
dim = '\033[2m'
warn = f'\033[1m{fg("indian_red_1c")}'
green = f'{bold}{fg("green")}'
blue = f'{bold}{fg("blue")}'
RED = f'{bold}{fg(52)}'
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
START_TIME2 = 0.0


def set_verbose(v: bool):
    """Sets the VERBOSE variable to True or False."""
    global VERBOSE
    VERBOSE = v


def get_verbose() -> bool:
    """Returns the current VERBOSE value."""
    return VERBOSE


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
    set_time2: bool = False,
    print_time: bool = False,
    print_time2: bool = False,
    print_time_first: bool = False,
    **kwargs,
):
    """
    ----- Special print function. -----
    indent: Allows easy indentation (2 spaces per 1 indent level) and easy styling.
    style: Style in which the text is printed. e.g. bold, dim, red, green, blue, etc. (as defined above), or any ANSI style code.
    verbose_in: Sets VERBOSE prior to printing, so that the print can be suppressed if desired.
    verbose_out: Sets VERBOSE after printing, so that the print can be suppressed if desired.
    set_time: Sets a timer to track elapsed time for subsequent prints.
    set_time2: Sets a second timer to track elapsed time for subsequent prints. e.g. when use set_time to plot time for each file processed, and set_time2 to plot time for all together.
    print_time: Prints the elapsed time since the last set_time.
    print_time2: Prints the elapsed time since the last set_time2.
    print_time_first: Prints the elapsed time since the last set_time, before printing the message.
    """
    global START_TIME
    global START_TIME2

    if verbose_in is not None:
        globals()['set_verbose'](verbose_in)

    if print_time_first:
        elapsed = time.time() - START_TIME
        print(f'[{elapsed:.1f} s]', end=' ')

    time_str = ''
    if print_time:
        elapsed = time.time() - START_TIME
        time_str = f' [{elapsed:.1f} s]'

    if set_time:
        START_TIME = time.time()

    if print_time2:
        elapsed = time.time() - START_TIME2
        time_str += f' [{elapsed:.1f} s]'

    if set_time2:
        START_TIME2 = time.time()

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
