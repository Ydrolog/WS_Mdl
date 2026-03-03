from filelock import FileLock as FL
from WS_Mdl.core.path import get_MdlN_Pa


def add_PKG(MdlN, str_PKG, iMOD5=False):
    """
    Adds a package (PKG) to the NAM file for the specified model (MdlN).
    str_PKG should be the exact line to add for the package, e.g. 'mvr6 mvr6.mvr6' (without quotes).
    Uses a file lock to ensure thread-safe file editing.
    """
    d_Pa = get_MdlN_Pa(MdlN, iMOD5=iMOD5)
    Pa_NAM = d_Pa['NAM_Mdl']

    lock = FL(Pa_NAM + '.lock')  # Create a file lock to prevent concurrent writes
    with lock, open(Pa_NAM, 'r+') as f:
        l_lines = f.readlines()
        l_lines[-1] = str_PKG + '\n'
        f.seek(0)
        f.truncate()
        for i in l_lines:
            f.write(i)
        f.write('END PACKAGES')
