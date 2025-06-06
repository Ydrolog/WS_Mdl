import shutil
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE

all_files = LD()

for f in all_files:
    try:
        shutil.copy2(f, f.replace("2018", "2019"))
    except:
        print(f, "not copied")