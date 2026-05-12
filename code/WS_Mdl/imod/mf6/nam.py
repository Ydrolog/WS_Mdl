from typing import TYPE_CHECKING

from filelock import FileLock as FL
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.imod.mf6.read import MF6_block_to_DF

if TYPE_CHECKING:
    import pandas as pd


def add_PKG(MdlN, str_PKG, iMOD5=False):
    """
    Backwards compatibility for add_Pkg. Renamed casue PKG doesn't follow the abbreviation rules of this project.
    """
    add_Pkg(MdlN, str_PKG, iMOD5=iMOD5)


def add_Pkg(MdlN, str_Pkg, iMOD5=False):
    """
    Adds a package (PKG) to the NAM file for the specified model (MdlN).
    str_PKG should be the exact line to add for the package, e.g. 'mvr6 mvr6.mvr6' (without quotes).
    Uses a file lock to ensure thread-safe file editing.
    """
    M = Mdl_N(MdlN, iMOD5=iMOD5)
    Pa_NAM = M.Pa.NAM_Mdl

    lock = FL(f'{Pa_NAM}.lock')  # Create a file lock to prevent concurrent writes
    with lock, open(Pa_NAM, 'r+') as f:
        l_lines = f.readlines()
        l_lines[-1] = str_Pkg + '\n'
        f.seek(0)
        f.truncate()
        for i in l_lines:
            f.write(i)
        f.write('END PACKAGES')


def DF_Pkgs(MdlN: str | Mdl_N) -> 'pd.DataFrame':
    """
    Retrieves the list of packages (PKGs) from the NAM file for the specified model (MdlN).
    Returns a list of package lines (excluding 'END PACKAGES').
    """
    M = Mdl_N(MdlN) if isinstance(MdlN, str) else MdlN

    return MF6_block_to_DF(M.Pa.NAM_Mdl, 'PACKAGES', has_header=False, names=['ftype', 'Rel_Pa', 'name'])


def l_Pkgs(MdlN: str | Mdl_N) -> list[str]:

    return sorted(DF_Pkgs(MdlN)['name'])
