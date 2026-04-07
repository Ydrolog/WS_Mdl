#!/usr/bin/env python
import sys

from WS_Mdl.imod.pop.wb import Diff_to_xlsx


def main():
    args = sys.argv[1:]
    if not args:
        print('Error: Missing mandatory argument.')
        sys.exit(1)
    Diff_to_xlsx(*args)


if __name__ == '__main__':
    main()
