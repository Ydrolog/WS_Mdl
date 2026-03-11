# ---------- Create Paths ----------
# I've converted to pathlib as it's more consistent. The only drawback is that it returns "WindowsPath" objects which is long and annoying, but you can use Pa_to_str to get rid of that problem.
import sys
from pathlib import Path

from .style import sprint

__all__ = ['REPO_ROOT', 'Pa_WS', 'Pa_RunLog', 'Pa_log_Out', 'Pa_log_Cfg', 'MdlN_Pa', 'imod_V', 'get_Mdl']


REPO_ROOT = Path(__file__).absolute().parents[3]

Pa_WS = REPO_ROOT
Pa_RunLog = Pa_WS / 'Mng/RunLog.xlsm'
Pa_log_Out = Pa_WS / 'Mng/log_Out.csv'
Pa_log_Cfg = Pa_WS / 'Mng/log_Cfg.csv'


def get_Mdl(MdlN: str):
    """Returns model alias (str part) for a given MdlN."""
    return ''.join([i for i in MdlN if i.isalpha()])


def imod_V(MdlN: str):
    """Returns the imod version used for a given MdlN."""

    Mdl = get_Mdl(MdlN)
    Pa_Sim = Pa_WS / f'models/{Mdl}/Sim/{MdlN}'

    try:
        names = {p.name for p in Pa_Sim.iterdir() if p.is_dir()}
        if 'modflow6' in names:
            return 'imod_python'
        elif 'GWF_1' in names:
            return 'imod5'
        else:
            sprint(
                f"Could not determine imod version from Sim/{MdlN} folder.\nProceeding with the assumption it's imod_python.",
                file=sys.stderr,
            )
            return 'imod_python'
    except FileNotFoundError:
        sprint(
            f"Could not determine imod version from Sim/{MdlN} folder.\nProceeding with the assumption that it's imod_python.",
            file=sys.stderr,
        )
        return 'imod_python'


def MdlN_Pa(MdlN: str, MdlN_B: str | bool | None = None, iMOD5: bool = None):
    """
    Returns a dictionary of paths related to the model number.
    - MdlN: model name string, e.g. 'NBr50'
    - MdlN_B: if provided, also include paths for the Baseline Sim of this model. Can be:
        - None or False (default): no Baseline Sim paths included.
        - True: automatically determine MdlN_B from RunLog based on MdlN.
        - str: use this string as MdlN_B directly.
    - iMOD5: if provided, forces the iMOD version. If None (default), it will be determined automatically based on folder structure.
    """
    # 666 If load time of MdlN_B is minimal, it should be don by default.
    if iMOD5 is None:
        iMOD5 = imod_V(MdlN) == 'imod5'

    def _replace_mdl_name(value, old_mdl: str, new_mdl: str):
        if value is None:
            return None
        if isinstance(value, Path):
            return Path(str(value).replace(old_mdl, new_mdl))
        if isinstance(value, str):
            return value.replace(old_mdl, new_mdl)
        return value

    def _MdlN_Pa_maker(MdlN):
        ## Non paths + General paths
        Mdl = get_Mdl(MdlN)  # Get model alias from MdlN

        d_Pa = {}
        d_Pa['imod_V'] = 'imod5' if iMOD5 else 'imod_python'
        d_Pa['WS'] = Pa_WS
        d_Pa['Mdl'] = Pa_WS / f'models/{Mdl}'
        d_Pa['Smk_temp'] = d_Pa['Mdl'] / 'code/snakemake/temp'
        d_Pa['In'] = d_Pa['Mdl'] / 'In'
        d_Pa['PoP'] = d_Pa['Mdl'] / 'PoP'

        d_Pa['code'] = Pa_WS / 'code'
        d_Pa['pixi'] = Pa_WS / 'pixi.toml'

        d_Pa['coupler_Exe'] = Pa_WS / r'software/iMOD5/IMC_2024.4/imodc.exe'
        d_Pa['MF6_DLL'] = d_Pa['coupler_Exe'].parent / './modflow6/libmf6.dll'
        d_Pa['MSW_DLL'] = d_Pa['coupler_Exe'].parent / './metaswap/MetaSWAP.dll'

        ## S Sim paths (grouped based on: pre-Sim, run, post-Sim)
        # Pre-Sim
        d_Pa['INI'] = d_Pa['Mdl'] / f'code/Mdl_Prep/Mdl_Prep_{MdlN}.ini'
        d_Pa['BAT'] = d_Pa['Mdl'] / f'code/Mdl_Prep/Mdl_Prep_{MdlN}.bat'
        d_Pa['PRJ'] = d_Pa['Mdl'] / f'In/PRJ/{MdlN}.prj'
        d_Pa['Smk'] = d_Pa['Mdl'] / f'code/snakemake/{MdlN}.smk'

        # Sim
        d_Pa['Sim'] = d_Pa['Mdl'] / 'Sim'  # Sim folder
        d_Pa['MdlN'] = d_Pa['Mdl'] / f'Sim/{MdlN}'
        d_Pa['MF6'] = d_Pa['MdlN'] / 'modflow6' if not iMOD5 else d_Pa['MdlN'] / 'GWF_1'
        d_Pa['MSW'] = d_Pa['MdlN'] / 'metaswap' if not iMOD5 else d_Pa['MdlN'] / 'GWF_1/MSWAPINPUT'
        d_Pa['TOML'] = d_Pa['MdlN'] / 'imod_coupler.toml'
        d_Pa['TOML_iMOD5'] = d_Pa['MdlN'] / f'{MdlN}.toml'
        d_Pa['Sim_In'] = d_Pa['MdlN'] / 'modflow6/imported_model' if not iMOD5 else d_Pa['MdlN'] / 'GWF_1/MODELINPUT'
        d_Pa['LST_Sim'] = d_Pa['MdlN'] / 'mfsim.lst'  # Sim LST file
        d_Pa['LST_Mdl'] = (
            d_Pa['Sim_In'] / 'imported_model.lst' if not iMOD5 else d_Pa['MdlN'] / f'GWF_1/{MdlN}.lst'
        )  # Mdl LST file
        d_Pa['NAM_Sim'] = d_Pa['MdlN'] / 'MFSIM.NAM'
        d_Pa['NAM_Mdl'] = (
            d_Pa['MdlN'] / 'modflow6/imported_model/imported_model.NAM'
            if not iMOD5
            else d_Pa['MdlN'] / f'GWF_1/{MdlN}.NAM'
        )
        d_Pa['Sim_Out'] = None if not iMOD5 else d_Pa['MdlN'] / 'GWF_1/MODELOUTPUT'
        d_Pa['SFR'] = (
            d_Pa['MdlN'] / f'modflow6/imported_model/{MdlN}.SFR6'
            if not iMOD5
            else d_Pa['MdlN'] / f'GWF_1/MODELINPUT/{MdlN}.SFR6'
        )

        # Post-run
        d_Pa['HD_Out_IDF'] = d_Pa['MdlN'] / 'GWF_1/MODELOUTPUT/HEAD'
        d_Pa['HD_Out_Bin'] = d_Pa['Sim_In'] / 'imported_model.hds' if not iMOD5 else d_Pa['HD_Out_IDF'] / 'HEAD.HED'
        d_Pa['DIS_GRB'] = d_Pa['Sim_In'] / f'{MdlN.upper()}.DIS6.grb' if iMOD5 else d_Pa['Sim_In'] / 'dis.dis.grb'
        d_Pa['PoP_Out_MdlN'] = d_Pa['PoP'] / 'Out' / MdlN
        d_Pa['MM'] = d_Pa['PoP_Out_MdlN'] / f'MM-{MdlN}.qgz'

        return d_Pa

    d_Pa = _MdlN_Pa_maker(MdlN)

    if MdlN_B:
        if MdlN_B is True:
            from WS_Mdl.core.log import to_Se

            MdlN_B_str = to_Se(MdlN)['B MdlN']  # Get MdlN_B from RunLog, based on MdlN
        elif isinstance(MdlN_B, str):
            MdlN_B_str = MdlN_B
        else:
            raise TypeError('MdlN_B should be None, False, True, or a model name string.')

        for k in list(d_Pa.keys()):
            if f'{k}_B' not in d_Pa:
                d_Pa[f'{k}_B'] = _replace_mdl_name(d_Pa[k], MdlN, MdlN_B_str)

    return d_Pa


class MdlN_PaView:
    """Makes MdlN_Pa dict keys accessible through MdlN, e.g. MdlN.Pa.INI instead of MdlN.Pa['INI']."""

    __slots__ = ('_d', '_MdlN')

    def __init__(self, MdlN: str, MdlN_B: str | None = None, iMOD5: bool | None = None):
        self._MdlN = MdlN
        self._d = MdlN_Pa(MdlN, MdlN_B=MdlN_B, iMOD5=iMOD5)

    @property
    def B(self):
        """Returns paths dictionary for the baseline simulation of this model."""
        from .log import get_B

        MdlN_B = get_B(self._MdlN)

        return MdlN_Pa(MdlN_B, iMOD5=(self._d['imod_V'] == 'imod5'))

    def _resolve_key(self, key: str) -> str:
        """Resolve legacy Pa_* keys to the current canonical key names."""
        if key in self._d:
            return key
        if key.startswith('Pa_'):
            key_no_prefix = key[3:]
            if key_no_prefix in self._d:
                return key_no_prefix
        raise KeyError(key)

    def __getattr__(self, name: str):
        # attribute-style access: Pa.INI -> dict["INI"]
        try:
            return self._d[self._resolve_key(name)]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key: str):
        return self._d[self._resolve_key(key)]

    def __repr__(self):
        # Show the underlying mapping directly so notebook print/display is readable.
        return repr(self._d)

    __str__ = __repr__

    def as_dict(self):
        """Returns a shallow copy as a plain dict."""
        return dict(self._d)

    def get(self, key: str, default=None):
        try:
            return self._d[self._resolve_key(key)]
        except KeyError:
            return default

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def values(self):
        return self._d.values()

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __contains__(self, key: str):
        if key in self._d:
            return True
        if isinstance(key, str) and key.startswith('Pa_'):
            return key[3:] in self._d
        return False
