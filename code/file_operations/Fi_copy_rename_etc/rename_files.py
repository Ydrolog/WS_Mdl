# Renames all files in the current directory.
import shutil
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE

all_files = LD()

for f in all_files:
    try:
        new_name = f.replace(".asc", "_NBr1.asc")
        os.rename(f, new_name)
        print(f"Renamed {f} to {new_name}")
    except:
        print(f"Failed to rename: {F}")