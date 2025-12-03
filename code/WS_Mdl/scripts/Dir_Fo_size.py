#!/usr/bin/env python
import os
from os.path import join as PJ


def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(PJ(dirpath, f))
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
        Dir = os.getcwd()

    sizes = {}
    for entry in os.scandir(Dir):
        if entry.is_dir():
            sizes[entry.name] = get_folder_size(entry.path)
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
