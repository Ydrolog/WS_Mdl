from pathlib import Path

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import Sep, bold, sprint
from WS_Mdl.imod.prj import PrSimP
from WS_Mdl.imod.sfr.prsimp import Pkgs_to_SFR_via_MVR, SFR_settings, connect_SFR_lines_to_MF6, create_SFR_lines


def Sim(
    M: Mdl_N,
    SFR: SFR_settings = True,
) -> None:
    """
    Prepares Sim Fis from In Fis following the process described in PrSimP's description (WS_Mdl.imod.prj.PrSimP).

    Set SFR = True to create an SFR package for that imod Sim. If SFR = True, the SFR_Config dataclass must be filled out and passed to the SFR argument. See SFR_Config for details.
    """
    sprint(Sep)
    sprint(f'----- Mdl_Prep: {M.MdlN} STARTING -----', style=bold, verbose_out=M.Sim.verbose)

    # %% Guard clause
    if (SFR is not False) and (not isinstance(SFR, SFR_settings)):
        raise ValueError(
            'SFR argument must be an instance of the SFR_settings dataclass with the appropriate parameters filled out or False. This is a Guard Clause.'
        )

    # %% Load PRJ & regrid it to Mdl area
    M.Sim_MF6, M.MSW_Mdl = timed_Exe(
        PrSimP,
        M,
        pre='--- imod PrSimP from PRJ file.',
        verbose_in=True,
        verbose_out=M.Sim.verbose,
        post='',
    )

    if SFR:
        # %% Create SFR Lines
        M.lines = timed_Exe(
            create_SFR_lines,
            Pa_GPkg=SFR.Pa_Gpkg,
            verbose=M.Sim.verbose,
            pre='--- SFRmaker\n -- Creating SFR lines.',
            verbose_in=True,
            verbose_out=M.Sim.verbose,
            post='',
        )

        # %% Connect SFR Lines to MF6 (writes files and connects them to NAM)
        sprint(' -- Connecting SFR lines to MF6.', verbose_in=True, verbose_out=M.Sim.verbose)
        M.Pa_SFR_OBS_In = (
            Path(SFR.Pa_OBS_In) if SFR.Pa_OBS_In is not None else None
        )  # 666 Those settings can also be moved to Mdl_N (mdl.py)
        M.Pa_Cond_A = Path(SFR.Pa_Cond_A)
        M.Pa_Cond_B = Path(SFR.Pa_Cond_B) if SFR.Pa_Cond_B is None else Path(SFR.Pa_Cond_A)
        M.SFR_OBS_all = SFR.OBS_all
        M.SFR_options = SFR.options
        M.SFR_minimum_reach_length = SFR.minimum_reach_length
        M.DF_reach = timed_Exe(
            connect_SFR_lines_to_MF6,
            M,
            debug_sfr=True,
        )

        # %% Connect DRN to SFR via MVR
        if SFR.connect_Pkgs:
            timed_Exe(
                Pkgs_to_SFR_via_MVR,
                M,
                Pkgs=SFR.connect_Pkgs,
                Pa_Shp=SFR.Pa_Shp_connect_Pkgs,
                pre='--- Connecting Pkgs to SFR via MVR.',
                verbose_in=True,
                verbose_out=M.Sim.verbose,
                post='',
            )

    sprint(f'----- Mdl_Prep: {M.MdlN} COMPLETED -----', style=bold, verbose_out=M.Sim.verbose)
