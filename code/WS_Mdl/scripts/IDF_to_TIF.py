#!/usr/bin/env python
import sys
from WS_Mdl.imod.idf import to_TIF


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <Pa_IDF1> [Pa_IDF2] [Pa_IDF3] ...')
        sys.exit(1)

    for Pa_IDF in sys.argv[1:]:  # Loop through all arguments after the script name
        to_TIF(Pa_IDF)


if __name__ == '__main__':
    main()
