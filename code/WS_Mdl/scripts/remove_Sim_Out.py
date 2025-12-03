#!/usr/bin/env python
import sys

from WS_Mdl.utils import remove_Sim_Out  # Adjust import as needed


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <MdlN1> [MdlN2] [MdlN3] ...')
        sys.exit(1)

    for MdlN in sys.argv[1:]:  # Loop through all arguments after the script name
        remove_Sim_Out(MdlN)


if __name__ == '__main__':
    main()
