#!/usr/bin/env python
import ast
import sys

from WS_Mdl.imod.pop.wb import Diff_to_xlsx


def _parse_arg(value):
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered == 'none':
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main():
    args = sys.argv[1:]
    if not args:
        print('Error: Missing mandatory argument.')
        sys.exit(1)

    # Separate positional and keyword arguments
    positional = []
    kwargs = {}
    for arg in args:
        if '=' in arg:
            key, val = arg.split('=', 1)
            kwargs[key] = _parse_arg(val)
        else:
            positional.append(_parse_arg(arg))

    Diff_to_xlsx(*positional, **kwargs)


if __name__ == '__main__':
    main()
