from datetime import datetime as DT
from pathlib import Path

from WS_Mdl.core import Mdl_N, Sep, bold, sprint
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.imod.prj import PrSimP
from WS_Mdl.imod.sfr.prsimp import Pkg_to_SFR_via_MVR, connect_SFR_lines_to_MF6, create_SFR_lines


def SFR_Mdl(
    MdlN: str,
    Pa_Cond_A: str | Path,
    Pa_Cond_B: str | Path = None,
    Pa_SFR_Gpkg: str | Path = None,
    Pa_MF6_DLL: str = None,
    Pa_MSW_DLL: str = None,
    Pa_SFR_OBS_In: str | Path = None,
    verbose: bool = False,
    add_DRN_to_SFR: bool = True,
    add_RIV_to_SFR: bool = True,
    Pa_Shp_DRN: str | Path = None,
    Pa_Shp_RIV: str | Path = None,
):
    """
    Prepares Sim Fis from In Fis.
    Ins need to be read and processed, then MF6 and MSW need to be coupled. Then Sim Ins can be written.
    """
    sprint(Sep)
    sprint(f'----- Mdl_Prep: {MdlN} -----', style=bold, verbose_out=verbose)

    # Create Mdl_N instance and enchance with params needed in following functions.
    sprint(' -- Loading MdlN parameters.', end='', verbose_in=True, verbose_out=verbose)
    M = Mdl_N(MdlN)
    # Dir_PRJ = PDN(Pa_PRJ)
    # d_INI = M.INI
    M.Xmin, M.Ymin, M.Xmax, M.Ymax, M.cellsize, M.N_R, M.N_C = M.Dmns
    M.SP_1st, M.SP_last = [DT.strftime(DT.strptime(M.INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']]
    # dx = dy = float(d_INI['CELLSIZE'])
    M.Pa.MF6_DLL = Pa_MF6_DLL if Pa_MF6_DLL else M.Pa.MF6_DLL  # If not specified, the default location will be used.
    M.Pa.MSW_DLL = Pa_MSW_DLL if Pa_MSW_DLL else M.Pa.MSW_DLL
    M.verbose = verbose
    sprint('🟢', verbose_in=True, verbose_out=verbose)

    # %% Load PRJ & regrid it to Mdl Aa
    sprint(' -- imod PrSimP from PRJ file.', verbose_in=True, verbose_out=verbose)
    M.Sim_MF6 = timed_Exe(PrSimP, M)

    # %% Create SFR Lines
    sprint(' -- SFRmaker - Creating SFR lines.', verbose_in=True, verbose_out=verbose)
    if Pa_SFR_Gpkg is None:
        Pa_SFR_Gpkg = M.Pa.In / f'SFR/{MdlN}/WBD_1ry_SW_NW_cleaned_{MdlN}.gpkg'
    M.lines = timed_Exe(create_SFR_lines, Pa_GPkg=Pa_SFR_Gpkg, verbose=M.verbose)

    # %% Connect SFR Lines to MF6 (writes files and connects them to NAM)
    sprint(' -- SFRmaker - Connecting SFR lines to MF6.', verbose_in=True, verbose_out=verbose)
    M.Pa_SFR_OBS_In = Path(Pa_SFR_OBS_In)
    M.Pa_Cond_A = Path(Pa_Cond_A)
    M.Pa_Cond_B = Path(Pa_Cond_A) if Pa_Cond_B is None else Path(Pa_Cond_B)
    M.DF_reach = timed_Exe(
        connect_SFR_lines_to_MF6,
        M,
        debug_sfr=True,
    )

    # %% Connect DRN to SFR via MVR
    sprint(' -- SFRmaker - Connecting DRN to SFR via MVR.', verbose_in=True, verbose_out=verbose)
    if add_DRN_to_SFR:
        timed_Exe(
            Pkg_to_SFR_via_MVR,
            M,
            Pkg='DRN',
            Pa_Shp=Pa_Shp_DRN,
        )

    # %% Connect RIV to SFR via MVR
    sprint(' -- SFRmaker - Connecting RIV to SFR via MVR.', verbose_in=True, verbose_out=verbose)
    if add_RIV_to_SFR:
        timed_Exe(
            Pkg_to_SFR_via_MVR,
            M,
            Pkg='RIV',
            Pa_Shp=Pa_Shp_RIV,
        )
