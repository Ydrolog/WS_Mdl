"""Normalize CSV files to minimal quoting after they are saved by Excel."""

import argparse
import csv
from pathlib import Path


def normalize_csv(path: Path) -> bool:
    """Rewrite *path* with quotes only where CSV syntax requires them."""
    original = path.read_bytes()
    if original.startswith(b'\xef\xbb\xbf'):
        encoding = 'utf-8-sig'
        text = original.decode(encoding)
    else:
        try:
            encoding = 'utf-8'
            text = original.decode(encoding)
        except UnicodeDecodeError:
            # Excel's plain "CSV (Comma delimited)" export uses the active
            # Windows code page; these log files historically use cp1252.
            encoding = 'cp1252'
            text = original.decode(encoding)

    rows = list(csv.reader(text.splitlines(keepends=True)))
    normalized_lines: list[str] = []

    class _LineWriter:
        def write(self, value: str) -> int:
            normalized_lines.append(value)
            return len(value)

    csv.writer(
        _LineWriter(),
        quoting=csv.QUOTE_MINIMAL,
        lineterminator='\n',
    ).writerows(rows)

    normalized = ''.join(normalized_lines).encode(encoding)
    if normalized == original:
        return False

    path.write_bytes(normalized)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()

    changed = False
    for path in args.files:
        if normalize_csv(path):
            print(f'Normalized CSV quoting: {path}')
            changed = True

    # A pre-commit hook should fail after changing files so the normalized
    # result can be reviewed and staged explicitly.
    return int(changed)


if __name__ == '__main__':
    raise SystemExit(main())
