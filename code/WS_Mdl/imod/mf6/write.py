from filelock import FileLock as FL
from WS_Mdl.core.path import MdlN_Pa
from WS_Mdl.core.style import sprint


def add_MVR_to_OPTIONS(Pa):
    """
    Opens a MODFLOW 6 input files (based on provided path), finds the OPTIONS block, and adds the MOVER option before the END OPTIONS line. Uses a file lock to ensure thread-safe file editing.

    Parameters
    ----------
    Pa : str
        Path to the MODFLOW 6 input file
    """
    lock = FL(f'{Pa}.lock')

    with lock:  # Acquire lock before editing file
        try:
            with open(Pa) as f:
                Lns = f.readlines()
            i = Lns.index('END OPTIONS\n')
            Lns[i] = '\tMOVER\nEND OPTIONS\n'
            with open(Pa, 'w') as f:
                f.writelines(Lns)
            sprint(f'🟢 - Added MOVER option to {Pa.name()}')
        except Exception as e:
            print(f'🔴 - Error adding MOVER option to {Pa.name()}: {e}')


def add_OBS_to_MF_In(str_OBS, PKG=None, MdlN=None, Pa=None, iMOD5=False):
    """
    Adds an OBS block to a MODFLOW 6 input file (to add to NAM, use utils.imod.py/add_OBS) (Pa). If Pa is not provided, it will be determined using MdlN and PKG.
    """

    if Pa is not None:
        Pa = Pa
    elif (MdlN is not None) and (PKG is not None):
        d_Pa = MdlN_Pa(MdlN, iMOD5=iMOD5)
        Pa = d_Pa['Sim_In'] / f'{MdlN}.{PKG}6'
    else:
        raise ValueError('Either Pa or both MdlN and PKG must be provided.')

    with open(Pa, 'r+') as f:
        l_Lns = f.readlines()
        try:
            i = next(i for i, ln in enumerate(l_Lns) if 'END OPTIONS' in ln.upper())
            l_1, l_2 = l_Lns[:i], l_Lns[i:]
            l_Lns = l_1 + [f'{str_OBS}\n'] + l_2
            f.seek(0)
            f.writelines(l_Lns)
            f.truncate()
            sprint(f'🟢 - Added OBS to {Pa}')
        except ValueError as e:
            print(f'🔴 - Failed:\n {e}')
