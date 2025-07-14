import shutil
import os
from os import listdir as LD
all_files = LD()

for f in all_files:
    try:
        shutil.copy2(f, f.replace("2018", "2019"))
    except:
        print(f, "not copied")