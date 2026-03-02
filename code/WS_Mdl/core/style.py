# ---------- Printing Section ----------

from colored import attr, fg

__all__ = ['Sep1', 'Sep2', 'style_reset', 'bold', 'dim', 'warn', 'CuCh', 'vprint', 'set_verbose', 'sprint']

style_reset = f'{attr("reset")}\033[0m'
bold = '\033[1m'
dim = '\033[2m'
warn = f'\033[1m{fg("indian_red_1c")}'
Sep_N = 100  # Number of characters in a separator line.
Sep1 = f'{fg(52)}{"-" * Sep_N}{attr("reset")}\n'  # Main separator - for script start and end.
Sep2 = f'{dim}{"-" * Sep_N}{attr("reset")}\n'  # Secondary separator - for sections within a script.


CuCh = {  # Stands for Custom Characters.
    '-': '🔴',  # negative
    '0': '🟡',  # neutral
    '+': '🟢',  # positive
    '=': '⚪️',  # no action required
    'x': '⚫️',  # already done
}

VERBOSE = True  # Use set_verbose to change this to False and get no information printed to the console.


def set_verbose(v: bool):
    """Sets the VERBOSE variable to True or False."""
    global VERBOSE
    VERBOSE = v


def sprint(indent: int = 0, *args, style: str, set_verbose: bool = None, **kwargs):
    """
    Special print function. Allows easy indentation (2 spaces per 1 indent level) and easy styling.
    Prints only if VERBOSE is True.
    """
    if set_verbose is not None:
        set_verbose(set_verbose)
    if VERBOSE:
        print(f'{style}{"  " * indent}', *args, f'{style_reset}', **kwargs)
