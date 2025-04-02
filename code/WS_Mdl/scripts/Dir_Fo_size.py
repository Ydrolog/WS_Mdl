#!/usr/bin/env python
def main():
    import os
    print('-'*100)
    Dir = input("Provide the directory for which to print all folder sizes. If you want the current directory, simply press Enter.\n")
    if Dir == '':
        Dir = os.getcwd()

    sizes = {d: sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, files in os.walk(os.path.join(Dir, d))
                    for f in files)
             for d in sorted(os.listdir(Dir), key=str.lower)
             if os.path.isdir(os.path.join(Dir, d))}

    for s in sizes:
        print(f"{s}: {sizes[s] / (1024 * 1024):.2f} MB")
    print('-'*100, '\n')