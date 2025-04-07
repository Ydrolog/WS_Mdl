#!/usr/bin/env python
import os

def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except (OSError, FileNotFoundError):
                pass
    return total

def main():
    print('-'*100)
    Dir = input("Provide the directory for which to print all folder sizes. If you want the current directory, simply press Enter.\n")
    if Dir == '':
        Dir = os.getcwd()

    sizes = {entry.name: get_folder_size(entry.path)
             for entry in os.scandir(Dir) if entry.is_dir()}

    for name, size in sorted(sizes.items()):
        print(f"{name}: {size / (1024 * 1024):.2f} MB")
    print('-'*100, '\n')