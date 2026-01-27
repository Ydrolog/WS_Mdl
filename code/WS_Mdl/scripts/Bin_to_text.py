#!/usr/bin/env python
import sys

from WS_Mdl.utils import Bin_to_text


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <binary_file_path>')
        sys.exit(1)

    bin_path = sys.argv[1]
    Bin_to_text(bin_path)


if __name__ == '__main__':
    main()
