#!/usr/bin/env python
import sys
from WS_Mdl.io.sim import S_from_B
from WS_Mdl.io.text import Sim_Cfg


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <MdlN1> [MdlN2] [MdlN3] ...')
        sys.exit(1)

    for MdlN in sys.argv[1:]:  # Loop through all arguments after the script name
        S_from_B(MdlN)
        Sim_Cfg(MdlN)


if __name__ == '__main__':
    main()
