#!/usr/bin/env python
"""Plot time series range from files with date patterns in their names."""

import argparse
import sys
from pathlib import Path

from WS_Mdl.utils import p_TS_range


def main():
    parser = argparse.ArgumentParser(
        description='Plot time series range from files with date patterns in their names.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python p_TS_range.py C:/data/rainfall --ending .IDF
  python p_TS_range.py C:/data/rainfall C:/data/temperature --ending .IDF
  python p_TS_range.py ./input_files --date-format %%Y-%%m-%%d --output my_plot.png
  python p_TS_range.py /path/to/files1 /path/to/files2 /path/to/files3 --ending .txt
        """,
    )

    parser.add_argument(
        'paths', nargs='+', help='One or more paths to directories containing files with dates in their names'
    )

    parser.add_argument('--ending', '-e', default='IDF', help='File extension to filter by (default: IDF)')

    parser.add_argument(
        '--date-format',
        '-d',
        default='%Y%m%d',
        help='Date format pattern for parsing dates from filenames (default: %%Y%%m%%d)',
    )

    parser.add_argument(
        '--output', '-o', default='TS_range.png', help='Output filename for the plot (default: TS_range.png)'
    )

    args = parser.parse_args()

    # Validate all paths exist
    valid_paths = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f'Warning: Path "{path_str}" does not exist. Skipping.')
            continue
        if not path.is_dir():
            print(f'Warning: Path "{path_str}" is not a directory. Skipping.')
            continue
        valid_paths.append(str(path))

    if not valid_paths:
        print('Error: No valid directories found.')
        sys.exit(1)

    try:
        print(f'Plotting time series range for files in {len(valid_paths)} path(s):')
        for path in valid_paths:
            print(f'  - {path}')
        print(f'File ending: {args.ending}')
        print(f'Date format: {args.date_format}')
        print(f'Output file: {args.output}')

        p_TS_range(l_Pa=valid_paths, ending=args.ending, date_format=args.date_format, Out_Fi=args.output)

        print('🟢 Plot completed!')

    except Exception as e:
        print(f'🔴 Error creating plot: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
