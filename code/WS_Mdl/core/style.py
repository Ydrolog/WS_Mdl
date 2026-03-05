# ---------- Printing Section ----------

from colored import attr, fg

__all__ = ['Sep', 'Sep_2', 'style_reset', 'bold', 'dim', 'warn', 'CuCh', 'set_verbose', 'sprint', 'sinput']

style_reset = f'{attr("reset")}\033[0m'
bold = '\033[1m'
dim = '\033[2m'
warn = f'\033[1m{fg("indian_red_1c")}'
MdlN_style = f'{bold}{fg("blue")}'
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


def set_verbose(v: bool):
    """Sets the VERBOSE variable to True or False."""
    global VERBOSE
    VERBOSE = v


def _is_Mdl_N(value):
    """Checks whether value is a WS_Mdl.core.mdl.Mdl_N instance."""
    try:
        from WS_Mdl.core.mdl import Mdl_N

        return isinstance(value, Mdl_N)
    except Exception:
        return value.__class__.__name__ == 'Mdl_N' and hasattr(value, 'MdlN')


def _fmt_arg(value, base_style: str = ''):
    """Formats args for sprint/sinput, highlighting MdlNs"""
    if _is_Mdl_N(value):
        return f'{MdlN_style}{value.MdlN}{style_reset}{base_style}'

    if isinstance(value, list):
        return '[' + ', '.join(str(_fmt_arg(v, base_style=base_style)) for v in value) + ']'

    if isinstance(value, tuple):
        items = ', '.join(str(_fmt_arg(v, base_style=base_style)) for v in value)
        if len(value) == 1:
            items += ','
        return f'({items})'

    if isinstance(value, set):
        if not value:
            return 'set()'
        return '{' + ', '.join(str(_fmt_arg(v, base_style=base_style)) for v in value) + '}'

    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f'{_fmt_arg(k, base_style=base_style)}: {_fmt_arg(v, base_style=base_style)}')
        return '{' + ', '.join(items) + '}'

    return value


def sprint(*args, indent: int = 0, style: str = '', set_verbose: bool = None, **kwargs):
    """
    Special print function. Allows easy indentation (2 spaces per 1 indent level) and easy styling.
    Prints only if VERBOSE is True.
    """
    if set_verbose is not None:
        globals()['set_verbose'](set_verbose)
    if VERBOSE:
        args_fmt = tuple(_fmt_arg(arg, base_style=style) for arg in args)
        prefix = f'{style}{"  " * indent}'

        if args_fmt:
            args_out = list(args_fmt)
            args_out[0] = f'{prefix}{args_out[0]}'
            args_out[-1] = f'{args_out[-1]}{style_reset}'
            print(*args_out, **kwargs)
        else:
            print(f'{prefix}{style_reset}', **kwargs)


def sinput(*args, indent: int = 0, style: str = '', sep: str = ' '):
    """
    Special input function. Allows easy indentation and easy styling.
    Always prompts, regardless of VERBOSE.
    """
    args_fmt = (_fmt_arg(arg, base_style=style) for arg in args)
    prompt = f'{style}{"  " * indent}{sep.join(str(arg) for arg in args_fmt)}{style_reset}'
    return input(prompt)
