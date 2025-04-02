import shutil
import os

all_files = os.listdir()

for f in all_files:
    try:
        shutil.copy2(f, f.replace("2018", "2019"))
    except:
        print(f, "not copied")