#!/usr/bin/env python
"""Open Mdl LST file for specified models."""

import sys

from WS_Mdl.core.path import MdlN_Pa
from WS_Mdl.io.text import o_VS


def main():
    if len(sys.argv) < 3:
        print(
            f'Usage: {sys.argv[0]} <key> <MdlN1> [MdlN2] [MdlN3] ...\nKeys need to be one of: {MdlN_Pa("dummy").keys()}'
        )
        sys.exit(1)

    try:
        o_VS(sys.argv[1], *sys.argv[2:])
    except Exception as e:
        print(f'Error opening files: {e}')
        sys.exit(1)
        print(f'Keys need to be one of: {MdlN_Pa("dummy").keys()}')


if __name__ == '__main__':
    main()
