# ---------- Create Paths ----------
# I've converted to pathlib as it's more consistent. The only drawback is that it returns "WindowsPath" objects which is long and annoying, but you can use Pa_to_str to get rid of that problem.
from pathlib import Path

from .style import vprint

__all__ = ['get_repo_root', 'Pa_WS', 'Pa_RunLog', 'Pa_log_Out', 'Pa_log_Cfg', 'get_MdlN_Pa', 'get_imod_V', 'get_Mdl']


def get_repo_root():  # Determine repository root dynamically at import time.
    """Return the repository root path.
    Tries to determine it from the file location, preserving subst drives (G:/).
    """
    try:
        # Use absolute() instead of resolve() to preserve subst drive letters (like G:)
        path = Path(__file__).absolute()
        # Assumes structure: <RepoRoot>/code/WS_Mdl/utils.py
        # So we go up 4 levels: path.py -> core -> WS_Mdl -> code -> RepoRoot
        root = path.parents[4 - 1]
        root_str = str(root).replace('\\', '/')
        return Path(root_str)
    except Exception as e:
        print(f'Error determining repository root: {e}')
        return None


Pa_WS = get_repo_root()
Pa_RunLog = Pa_WS / 'Mng/RunLog.xlsx'
Pa_log_Out = Pa_WS / 'Mng/log_Out.csv'
Pa_log_Cfg = Pa_WS / 'Mng/log_Cfg.csv'


def Pa_to_str(a_list: list[Path]):
    """Helper function to convert a list of paths to a list of strings (for subprocess calls)."""
    return [str(p) for p in a_list]


def get_Mdl(MdlN: str):
    """Returns model alias for a given MdlN."""
    return ''.join([i for i in MdlN if i.isalpha()])


def get_imod_V(MdlN: str):
    """Returns the imod version used for a given MdlN."""

    Mdl = get_Mdl(MdlN)
    Pa_Sim = Pa_WS / f'models/{Mdl}/Sim/{MdlN}'

    try:
        if 'modflow6' in [p.name for p in Pa_Sim.iterdir() if p.is_dir()]:
            return 'imod_python'
        elif 'GWF_1' in [p.name for p in Pa_Sim.iterdir() if p.is_dir()]:
            return 'imod5'
        else:
            print(
                f"Could not determine imod version Sim/{MdlN} folder doesn't exist, or it's structure has been modified.\n Proceeding with the assumption that it's imod_python."
            )
            return 'imod_python'
    except FileNotFoundError:
        print(
            f"Could not determine imod version Sim/{MdlN} folder doesn't exist, or it's structure has been modified.\n Proceeding with the assumption that it's imod_python."
        )
        return 'imod_python'


def get_MdlN_Pa(MdlN: str, MdlN_B: str = None, iMOD5: bool = None):
    """
    *** Improved get_MdlN_paths. ***
    - Doesn't read RunLog, unless B is set to True. Thus it's much faster.
    - Returns a dictionary of useful objects (mainly paths, but also Mdl, MdlN) for a given MdlN. Those need to then be passed to arguments, e.g.:
        d_Pa = get_MdlN_Pa(MdlN)
        Pa_INI = d_Pa['Pa_INI'].

    This function has been modified since NBr32, to support imod python's folder/file structure. If you need to use the old folder structure, set iMOD5=True.
    """
    if iMOD5 is None:
        if get_imod_V(MdlN) == 'imod5':
            iMOD5 = True
        elif get_imod_V(MdlN) == 'imod_python':
            iMOD5 = False
        else:
            iMOD5 = False
            vprint(
                f"🔴 - Couldn't determine imod version from Sim/{MdlN} folder. Proceeding assuming it's imod_python. Provide iMOD5=True if you want to proceed with iMOD5's structure instead."
            )

    def _MdlN_Pa_maker(MdlN):
        ## Non paths + General paths
        Mdl = get_Mdl(MdlN)  # Get model alias from MdlN

        d_Pa = {}
        d_Pa['imod_V'] = 'imod5' if iMOD5 else 'imod_python'
        d_Pa['Mdl'] = Mdl
        d_Pa['MdlN'] = MdlN
        d_Pa['Pa_Mdl'] = Pa_WS / f'models/{Mdl}'
        d_Pa['Smk_temp'] = d_Pa['Pa_Mdl'] / 'code/snakemake/temp'
        d_Pa['In'] = d_Pa['Pa_Mdl'] / 'In'
        d_Pa['PoP'] = d_Pa['Pa_Mdl'] / 'PoP'

        d_Pa['code'] = Pa_WS / 'code'
        d_Pa['pixi'] = Pa_WS / 'pixi.toml'

        d_Pa['coupler_Exe'] = Pa_WS / r'software/iMOD5/IMC_2024.4/imodc.exe'
        d_Pa['MF6_DLL'] = d_Pa['coupler_Exe'].parent / './modflow6/libmf6.dll'
        d_Pa['MSW_DLL'] = d_Pa['coupler_Exe'].parent / './metaswap/MetaSWAP.dll'

        ## S Sim paths (grouped based on: pre-Sim, run, post-Sim)
        # Pre-Sim
        d_Pa['INI'] = d_Pa['Pa_Mdl'] / f'code/Mdl_Prep/Mdl_Prep_{MdlN}.ini'
        d_Pa['BAT'] = d_Pa['Pa_Mdl'] / f'code/Mdl_Prep/Mdl_Prep_{MdlN}.bat'
        d_Pa['PRJ'] = d_Pa['Pa_Mdl'] / f'In/PRJ/{MdlN}.prj'
        d_Pa['Smk'] = d_Pa['Pa_Mdl'] / f'code/snakemake/{MdlN}.smk'

        # Sim
        d_Pa['Sim'] = d_Pa['Pa_Mdl'] / 'Sim'  # Sim folder
        d_Pa['Pa_MdlN'] = d_Pa['Pa_Mdl'] / f'Sim/{MdlN}'
        d_Pa['MF6'] = d_Pa['Pa_MdlN'] / 'modflow6' if not iMOD5 else d_Pa['Pa_MdlN'] / 'GWF_1'
        d_Pa['MSW'] = d_Pa['Pa_MdlN'] / 'metaswap' if not iMOD5 else d_Pa['Pa_MdlN'] / 'GWF_1/MSWAPINPUT'
        d_Pa['TOML'] = d_Pa['Pa_MdlN'] / 'imod_coupler.toml'
        d_Pa['TOML_iMOD5'] = d_Pa['Pa_MdlN'] / f'{MdlN}.toml'
        d_Pa['Sim_In'] = (
            d_Pa['Pa_MdlN'] / 'modflow6/imported_model' if not iMOD5 else d_Pa['Pa_MdlN'] / 'GWF_1/MODELINPUT'
        )
        d_Pa['LST_Sim'] = d_Pa['Pa_MdlN'] / 'mfsim.lst'  # Sim LST file
        d_Pa['LST_Mdl'] = (
            d_Pa['Sim_In'] / 'imported_model.lst' if not iMOD5 else d_Pa['Pa_MdlN'] / f'GWF_1/{MdlN}.lst'
        )  # Mdl LST file
        d_Pa['NAM_Sim'] = d_Pa['Pa_MdlN'] / 'MFSIM.NAM'
        d_Pa['NAM_Mdl'] = (
            d_Pa['Pa_MdlN'] / 'modflow6/imported_model/imported_model.NAM'
            if not iMOD5
            else d_Pa['Pa_MdlN'] / f'GWF_1/{MdlN}.NAM'
        )
        d_Pa['Sim_Out'] = None if not iMOD5 else d_Pa['Pa_MdlN'] / 'GWF_1/MODELOUTPUT'
        d_Pa['SFR'] = (
            d_Pa['Pa_MdlN'] / f'modflow6/imported_model/{MdlN}.SFR6'
            if not iMOD5
            else d_Pa['Pa_MdlN'] / f'GWF_1/MODELINPUT/{MdlN}.SFR6'
        )

        # Post-run
        d_Pa['Out_HD'] = d_Pa['Pa_MdlN'] / 'GWF_1/MODELOUTPUT/HEAD'
        d_Pa['Out_HD_Bin'] = d_Pa['Sim_In'] / 'imported_model.hds' if not iMOD5 else d_Pa['Out_HD'] / 'HEAD.HED'
        d_Pa['DIS_GRB'] = d_Pa['Sim_In'] / f'{MdlN.upper()}.DIS6.grb' if iMOD5 else d_Pa['Sim_In'] / 'dis.dis.grb'
        d_Pa['PoP_Out_MdlN'] = d_Pa['PoP'] / 'Out' / MdlN
        d_Pa['MM'] = d_Pa['PoP_Out_MdlN'] / f'MM-{MdlN}.qgz'

        if MdlN_B:  ## B Sim paths
            for k in list(d_Pa.keys()):
                if f'{k}_B' not in d_Pa:
                    d_Pa[f'{k}_B'] = d_Pa[k].replace(MdlN, MdlN_B)
        return d_Pa

    d_Pa = _MdlN_Pa_maker(MdlN) if MdlN_B is None else _MdlN_Pa_maker(MdlN) | _MdlN_Pa_maker(MdlN_B)
    return d_Pa
