#!/usr/bin/env python
"""Open Mdl LST file for specified models."""

import sys

from WS_Mdl.io.text import o_


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <MdlN1> [MdlN2] [MdlN3] ...')
        sys.exit(1)

    for MdlN in sys.argv[1:]:  # Loop through all arguments after the script name
        o_('LST_Mdl', MdlN)


if __name__ == '__main__':
    main()
