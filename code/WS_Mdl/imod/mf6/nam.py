from filelock import FileLock as FL
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.path import MdlN_PaView


def add_PKG(MdlN, str_PKG, iMOD5=False):
    """
    Adds a package (PKG) to the NAM file for the specified model (MdlN).
    str_PKG should be the exact line to add for the package, e.g. 'mvr6 mvr6.mvr6' (without quotes).
    Uses a file lock to ensure thread-safe file editing.
    """
    M = Mdl_N(MdlN)
    Pa = M.Pa if iMOD5 == (M.V == 'imod5') else MdlN_PaView(MdlN, iMOD5=iMOD5)
    Pa_NAM = Pa.NAM_Mdl

    lock = FL(f'{Pa_NAM}.lock')  # Create a file lock to prevent concurrent writes
    with lock, open(Pa_NAM, 'r+') as f:
        l_lines = f.readlines()
        l_lines[-1] = str_PKG + '\n'
        f.seek(0)
        f.truncate()
        for i in l_lines:
            f.write(i)
        f.write('END PACKAGES')
