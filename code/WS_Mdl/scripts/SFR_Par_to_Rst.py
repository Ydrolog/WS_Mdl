#!/usr/bin/env python
import sys
from WS_Mdl.geo import SFR_Par_to_Rst

def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <MdlN> <Par>')
        print('Example: SFR_Par_to_Rst NBr10 rtp')
        sys.exit(1)

    MdlN = sys.argv[1]
    Par = sys.argv[2]
    
    SFR_Par_to_Rst(MdlN, Par)

if __name__ == '__main__':
    main()
