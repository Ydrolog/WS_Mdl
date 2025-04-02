# Renames all files in the current directory.
import shutil
import os

all_files = os.listdir()

for f in all_files:
    try:
        new_name = f.replace(".asc", "_NBr1.asc")
        os.rename(f, new_name)
        print(f"Renamed {f} to {new_name}")
    except:
        print(f"Failed to rename: {F}")