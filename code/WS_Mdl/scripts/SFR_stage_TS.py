#!/usr/bin/env python
import sys

from WS_Mdl.utils_imod import SFR_stage_TS  # Adjust import as needed


def main():
    _, MdlN_A, MdlN_B, layer = sys.argv
    SFR_stage_TS(MdlN_A, MdlN_B, int(layer))


if __name__ == '__main__':
    main()
