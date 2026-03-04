#!/usr/bin/env python
from pathlib import Path


def get_folder_size(path):
    total = 0
    stack = [Path(path)]

    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, FileNotFoundError):
            continue

        for entry in entries:
            try:
                if entry.is_dir() and not entry.is_symlink():
                    stack.append(entry)
                elif entry.is_file():
                    total += entry.stat().st_size
            except (OSError, FileNotFoundError):
                pass

    return total


def main(Dir=None, sort_by='size'):
    print('-' * 100)
    if Dir is None:
        Dir = input(
            'Provide the directory for which to print all folder sizes. If you want the current directory, simply press Enter.\n'
        )
    if Dir == '':
        Dir = Path.cwd()
    else:
        Dir = Path(Dir)

    sizes = {}
    for entry in Dir.iterdir():
        if entry.is_dir():
            sizes[entry.name] = get_folder_size(entry)
        elif entry.is_file():
            try:
                sizes[entry.name] = entry.stat().st_size
            except (OSError, FileNotFoundError):
                pass

    if sort_by == 'size':
        items = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
    else:
        items = sorted(sizes.items())

    for name, size in items:
        print(f'{name}: {size / (1024 * 1024):.2f} MB')
    print('-' * 100, '\n')
