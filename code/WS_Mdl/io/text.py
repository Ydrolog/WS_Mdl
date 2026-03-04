import subprocess as sp

from WS_Mdl.core.path import MdlN_Pa
from WS_Mdl.core.style import Sep, sprint
from WS_Mdl.utils import sprint


def o_(key, *l_MdlN, Pa=r'C:\Program Files\Notepad++\notepad++.exe'):
    """Opens files at default locations, as specified by MdlN_Pa()."""
    if key not in MdlN_Pa('NBr1').keys():
        raise ValueError(f'\nInvalid key: {key}.\nValid keys are: {", ".join(MdlN_Pa("NBr1").keys())}')
        return

    sprint(Sep)
    sprint(f'\nOpening {key} file(s) for specified run(s) with the default program.\n')
    sprint(
        f"It's assumed that Notepad++ is installed in: {Pa}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument (Pa) to this function.\n"
    )

    l_Pa = [MdlN_Pa(MdlN)[key] for MdlN in l_MdlN]

    for f in l_Pa:
        sp.Popen([Pa] + [f])
        sprint(f'🟢 - {f}')
    sprint(Sep)


def o_VS(key, *l_MdlN, Pa='code'):
    """Opens files at default locations with VS Code, as specified by MdlN_Pa()."""
    if key not in MdlN_Pa('NBr1').keys():
        raise ValueError(f'\nInvalid key: {key}.\nValid keys are: {", ".join(MdlN_Pa("NBr1").keys())}')
        return

    sprint(Sep)
    sprint(f'\nOpening {key} file(s) for specified run(s) with VS Code.\n')
    sprint(
        "It's assumed that VS Code is accessible via the 'code' command.\nIf that's not True, provide the correct path to VS Code as the last argument (Pa) to this function.\n"
    )

    l_Pa = [MdlN_Pa(MdlN)[key] for MdlN in l_MdlN]

    for f in l_Pa:
        sp.Popen([Pa, f], shell=True)
        sprint(f'🟢 - {f}')
    sprint(Sep)


def Sim_Cfg(*l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    sprint()
    sprint(
        f"Opening all configuration files for specified runs with the default program.\nIt's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['Smk', 'BAT', 'INI', 'PRJ']
    l_paths = [MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]
    sp.Popen([Pa_NP] + l_files)
    for f in l_files:
        sprint(f'🟢 - {f}')


def o_LSTs(*l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    sprint()
    sprint('Opening LST files (Mdl+Sim) for specified runs with the default program.\n')
    sprint(
        f"It's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['LST_Sim', 'LST_Mdl']
    l_paths = [MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for paths in l_paths for k in l_keys]

    for f in l_files:
        sp.Popen([Pa_NP] + [f])
        sprint(f'🟢 - {f}')


def o_NAMs(*l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    sprint()
    sprint('Opening NAM files (Mdl+Sim) for specified runs with the default program.\n')
    sprint(
        f"It's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['NAM_Sim', 'NAM_Mdl']
    l_paths = [MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for paths in l_paths for k in l_keys]

    for f in l_files:
        sp.Popen([Pa_NP] + [f])
        sprint(f'🟢 - {f}')
