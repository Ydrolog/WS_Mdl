import sys
import subprocess
from pathlib import Path

"""Runs "dvc add" for all files located directly in a provided path that match a patern (e.g. .idf). If path is not provided, the current path is used."""


def main():
    if len(sys.argv) < 2:
        print('Usage: DVC_add_pattern <pattern> [target_dir]')
        sys.exit(1)

    pattern = sys.argv[1]
    target_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()

    print(f"Looking for '{pattern}' in: {target_dir.resolve()}")

    for path in target_dir.rglob(pattern):
        print(f'Adding to DVC: {path}')
        subprocess.run(['dvc', 'add', str(path)], check=True)
