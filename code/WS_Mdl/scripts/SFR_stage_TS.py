#!/usr/bin/env python
import sys

from WS_Mdl.imod.pop.sfr import stage_TS


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <MdlN> <MdlN_RIV> [N_system_RIV] [N_system_DRN] ...')
        sys.exit(1)
    else:
        stage_TS(*sys.argv[1:3], int(sys.argv[3]), int(sys.argv[4]))


if __name__ == '__main__':
    main()
