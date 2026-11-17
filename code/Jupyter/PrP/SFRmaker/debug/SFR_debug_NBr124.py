# %% [markdown] 0. Imports
# ## 0. Imports
from pathlib import Path

import WS_Mdl.core.df  # noqa: F401
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import sprint
from WS_Mdl.imod.prj import PrSimP
from WS_Mdl.imod.sfr.prsimp import SFR_settings, connect_SFR_lines_to_MF6, create_SFR_lines

# %% [markdown] 1. Options
# ## 1. Options
MdlN = 'NBr124'
M = Mdl_N(MdlN)
MdlN_SFR_GPkg = 'NBr124'

# SFR Options
Pa_SFR_GPkg = M.Pa.In / f'SFR/{MdlN_SFR_GPkg}/WBD_detail_SW_NW_cleaned_{MdlN_SFR_GPkg}.gpkg'
Pa_SW_Cond_A = M.Pa.WS / r'models\NBr\In\RIV\RIV_Cond_DETAILWATERGANGEN_NBr1.IDF'
Pa_SW_Cond_B = M.Pa.WS / r'models\NBr\In\RIV\RIV_Cond_DRN_NBr1.IDF'
Pa_SFR_OBS_In = M.Pa.In / 'OBS/SFR/NBr73/NBr73_SFR_OBS_Pnt.csv'
SFR_connect_Pkgs = ('DRN', 'RIV')
Pa_Shp_catchment = M.Pa.PoP / r'common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp'
SFR_OBS_all = ['sfr', 'downstream-flow', 'inflow', 'stage', 'from-mvr']
SFR_options = [
    f'OBS6 FILEIN {M.Pa.Sim_In / (MdlN + ".SFR6.obs")}',
    f'BUDGET FILEOUT Out/{MdlN}.SFR6.cbc',  # 666 Remove this if it doesn't contain any useful info
    # 'AUXILIARY line_id',
    # f'STAGE FILEOUT SFR_Stg_{MdlN}.bin', # unnecessary because we have OBS for all reaches
    f'PACKAGE_CONVERGENCE FILEOUT Out/SFR_convergence_{MdlN}.CSV',
    'SAVE_FLOWS',
    'STORAGE',
]
SFR_one_reach_per_cell: bool = True

# %% [markdown] 2. Run Prep
# ## 2. Run Prep
SFR = SFR_settings(
    Pa_Cond_A=Pa_SW_Cond_A,
    Pa_Cond_B=Pa_SW_Cond_B,
    Pa_Gpkg=Pa_SFR_GPkg,
    Pa_OBS_In=Pa_SFR_OBS_In,
    connect_Pkgs=SFR_connect_Pkgs,
    Pa_Shp_connect_Pkgs=Pa_Shp_catchment,
    OBS_all=SFR_OBS_all,
    options=SFR_options,
    one_reach_per_cell=SFR_one_reach_per_cell,
    # Stg_Init            = Pa_SFR_Stg_Init
)

if (SFR is not False) and (not isinstance(SFR, SFR_settings)):
    raise ValueError(
        'SFR argument must be an instance of the SFR_settings dataclass with the appropriate parameters filled out or False. This is a Guard Clause.'
    )

# %% [markdown] 3. PrSimP
# ## 3. PrSimP
M.Sim_MF6, M.MSW_Mdl = timed_Exe(
    PrSimP,
    M,
    pre='--- imod PrSimP from PRJ file.\n',
    verbose_in=True,
    verbose_out=M.Sim.verbose,
    post='--- 🟢🟢🟢',
)

# %% [markdown] 4. create_SFR_lines
# ## 4. create_SFR_lines
Pa_GPkg = Pa_SFR_GPkg
verbose = True
debug_sfr = True


Pa_GPkg = Path(Pa_GPkg)

from pathlib import Path

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.imod.prj import PrSimP
from WS_Mdl.imod.sfr.prsimp import SFR_settings

M.lines = timed_Exe(
    create_SFR_lines,
    Pa_GPkg=SFR.Pa_Gpkg,
    verbose=M.Sim.verbose,
    pre='--- SFRmaker\n -- Creating SFR lines.\n',
    verbose_in=True,
    verbose_out=M.Sim.verbose,
    post=' -- 🟢🟢',
)


# %% [markdown] 5. connect_SFR_lines_to_MF6
# ## 5. connect_SFR_lines_to_MF6
sprint(' -- Connecting SFR lines to MF6.', verbose_in=True, verbose_out=M.Sim.verbose)
M.Pa_SFR_OBS_In = (
    Path(SFR.Pa_OBS_In) if SFR.Pa_OBS_In is not None else None
)  # 666 Those settings can also be moved to Mdl_N (mdl.py)
M.Pa_Cond_A = Path(SFR.Pa_Cond_A)
M.Pa_Cond_B = Path(SFR.Pa_Cond_B) if SFR.Pa_Cond_B is None else Path(SFR.Pa_Cond_A)
M.SFR_OBS_all = SFR.OBS_all
M.SFR_options = SFR.options
M.SFR_minimum_reach_length = SFR.minimum_reach_length
M.SFR_one_reach_per_cell = SFR.one_reach_per_cell
M.SFR_consolidate_conductance = SFR.consolidate_conductance
M.SFR_Stg_Init = SFR.Stg_Init
M.DF_reach = timed_Exe(
    connect_SFR_lines_to_MF6,
    M,
    debug_sfr=True,
    post=' -- Connecting SFR lines to MF6.🟢🟢\n',
)
sprint('--- 🟢🟢🟢', verbose_in=True, verbose_out=M.Sim.verbose)


# %% [markdown] 6. Pkgs_to_SFR_via_MVR
# ## 6. Pkgs_to_SFR_via_MVR
from WS_Mdl.imod.sfr.prsimp import Pkgs_to_SFR_via_MVR

if SFR.connect_Pkgs:
    timed_Exe(
        Pkgs_to_SFR_via_MVR,
        M,
        Pkgs=SFR.connect_Pkgs,
        Pa_Shp=SFR.Pa_Shp_connect_Pkgs,
        pre='--- Connecting Pkgs to SFR via MVR.\n',
        verbose_in=True,
        verbose_out=M.Sim.verbose,
        post='--- 🟢🟢🟢',
    )

# %%
