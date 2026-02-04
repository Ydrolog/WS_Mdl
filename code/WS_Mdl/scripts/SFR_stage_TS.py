#!/usr/bin/env python
import sys

from WS_Mdl.utils_imod import SFR_stage_TS  # Adjust import as needed


def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <MdlN> <MdlN_RIV> [N_system_RIV] [N_system_DRN] ...')
        sys.exit(1)
    else:
        SFR_stage_TS(*sys.argv[1:3], int(sys.argv[3]), int(sys.argv[4]))


if __name__ == '__main__':
    main()
