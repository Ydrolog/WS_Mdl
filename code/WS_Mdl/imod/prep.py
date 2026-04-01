from dataclasses import dataclass
from pathlib import Path

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import Sep, bold, sprint
from WS_Mdl.imod.prj import PrSimP
from WS_Mdl.imod.sfr.prsimp import Pkgs_to_SFR_via_MVR, connect_SFR_lines_to_MF6, create_SFR_lines


@dataclass
class SFR_settings:
    Pa_Gpkg: str | Path  # Main SFR geopackage. # 666 details about required columns need to be written.
    Pa_Cond_A: str | Path  # Primary conductance file for SFR. This is required.
    Pa_Cond_B: str | Path | None = (
        None  # Optional secondary conductance file for SFR. If not specified, only the primary conductance file will be used.
    )
    Pa_OBS_In: str | Path | None = None  # OBS for SFR
    connect_Pkgs: tuple = ()  # Option to connect DRN to SFR via MVR.
    Pa_Shp_connect_Pkgs: str | Path | None = (
        None  # Shapefile containing the outer boundaries of the DRN package cells to be connected to the nearest SFR cells.
    )


def Sim(
    MdlN: str,
    verbose: bool = False,
    SFR: bool = True,
    Pa_MF6_DLL: str = None,
    Pa_MSW_DLL: str = None,
):
    """
    Prepares Sim Fis from In Fis following the process described in PrSimP's description (WS_Mdl.imod.prj.PrSimP).

    Set SFR = True to create an SFR package for that imod Sim. If SFR = True, the SFR_Config dataclass must be filled out and passed to the SFR argument. See SFR_Config for details.
    """
    sprint(Sep)
    sprint(f'----- Mdl_Prep: {MdlN} -----', style=bold, verbose_out=verbose)

    # Create Mdl_N instance and enchance with params needed in following functions.
    sprint('--- Loading MdlN parameters.', end='', verbose_in=True, verbose_out=verbose, set_time=True)
    M = Mdl_N(MdlN)
    M.Pa.MF6_DLL = Pa_MF6_DLL if Pa_MF6_DLL else M.Pa.MF6_DLL  # If not specified, the default location will be used.
    M.Pa.MSW_DLL = Pa_MSW_DLL if Pa_MSW_DLL else M.Pa.MSW_DLL
    M.verbose = verbose
    sprint('🟢', verbose_in=True, verbose_out=verbose, print_time=True)

    # %% Load PRJ & regrid it to Mdl Aa
    sprint('--- imod PrSimP from PRJ file.', verbose_in=True, verbose_out=verbose)
    M.Sim_MF6, M.MSW_Mdl = timed_Exe(PrSimP, M)

    if SFR:
        # %% Create SFR Lines
        sprint('--- SFRmaker\n -- Creating SFR lines.', verbose_in=True, verbose_out=verbose)
        M.lines = timed_Exe(create_SFR_lines, Pa_GPkg=SFR.Pa_Gpkg, verbose=M.verbose)

        # %% Connect SFR Lines to MF6 (writes files and connects them to NAM)
        sprint(' -- Connecting SFR lines to MF6.', verbose_in=True, verbose_out=verbose)
        M.Pa_SFR_OBS_In = Path(SFR.Pa_OBS_In)
        M.Pa_Cond_A = Path(SFR.Pa_Cond_A)
        M.Pa_Cond_B = Path(SFR.Pa_Cond_B) if SFR.Pa_Cond_B is None else Path(SFR.Pa_Cond_B)
        M.DF_reach = timed_Exe(
            connect_SFR_lines_to_MF6,
            M,
            debug_sfr=True,
        )

        # %% Connect DRN to SFR via MVR
        if SFR.connect_Pkgs:
            sprint(f' -- Connecting Pkgs ({SFR.connect_Pkgs}) to SFR via MVR.', verbose_in=True, verbose_out=verbose)
            timed_Exe(
                Pkgs_to_SFR_via_MVR,
                M,
                Pkgs=SFR.connect_Pkgs,
                Pa_Shp=SFR.Pa_Shp_connect_Pkgs,
            )
