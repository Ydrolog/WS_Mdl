#!/usr/bin/env python
import sys
from WS_Mdl import S_from_B  # Adjust import as needed

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <MdlN>")
        sys.exit(1)

    MdlN = sys.argv[1]
    S_from_B(MdlN)

if __name__ == "__main__":
    main()
