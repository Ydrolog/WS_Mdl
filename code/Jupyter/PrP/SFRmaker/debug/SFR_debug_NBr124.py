# %% [markdown] 0. Imports
# ## 0. Imports
from pathlib import Path

import WS_Mdl.core.df  # noqa: F401
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import sprint
from WS_Mdl.imod.prj import PrSimP
from WS_Mdl.imod.sfr.prsimp import SFR_settings

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

import sfrmaker as sfr

Pa_GPkg = Path(Pa_GPkg)
import sys

import geopandas as gpd
import pandas as pd

# %% Load GPkg
sprint(' -- Create SFR lines.', verbose_in=True, verbose_out=verbose)
GDF = gpd.read_file(Pa_GPkg)

# %% Ensure slope
sprint('  - Ensure slope.', verbose_in=True, verbose_out=verbose, set_time=True)
if debug_sfr:
    from tabulate import tabulate

    sprint(
        tabulate(
            GDF[['ID', 'Elv_UStr', 'Elv_DStr']].describe(include='all'),
            headers='keys',
            tablefmt='grid',
            showindex=True,
        ),
        verbose_in=True,
        verbose_out=verbose,
        indent=0,
    )

# %% Check NaN counts
NaN_count_UStr = GDF['Elv_UStr'].isna().sum()
NaN_count_DStr = GDF['Elv_DStr'].isna().sum()
if NaN_count_UStr > 0 or NaN_count_DStr > 0:
    sys.exit(
        f'ERROR: There are {NaN_count_UStr} NaN values in Elv_UStr and {NaN_count_DStr} NaN values in Elv_DStr. Please fill these values before proceeding.'
    )

# %% Enumerate segments by slope
GDF_Elv = GDF[['ID', 'Elv_UStr', 'Elv_DStr', 'DStr_code', 'DStr_ID']].copy()
GDF_Elv['Diff'] = GDF_Elv['Elv_UStr'] - GDF_Elv['Elv_DStr']

N_UStr_LT_DStr = int((GDF_Elv['Diff'] < 0).sum())
N_UStr_Eq_DStr = int((GDF_Elv['Diff'] == 0).sum())
N_UStr_GT_DStr = int((GDF_Elv['Diff'] > 0).sum())

sprint(
    f'    There are:\n\t{N_UStr_LT_DStr:6} segments where Elv_UStr < Elv_DStr\n\t{N_UStr_Eq_DStr:6} segments where Elv_UStr = Elv_DStr\n\t{N_UStr_GT_DStr:6} segments where Elv_UStr > Elv_DStr\n\t{GDF.shape[0]:6} segments in total.',
    verbose_in=True,
    verbose_out=verbose,
)

# %% Identify problematic Elv segments
sprint(
    "It's assumed SFRmaker works where Elv_UStr >= Elv_DStr. Only segments where Elv_UStr < Elv_DStr will be adjusted.",
    verbose_in=True,
    verbose_out=verbose,
    indent=3,
)

if debug_sfr:
    sprint('\tIncline segments:', indent=3, verbose_in=True, verbose_out=verbose)
    sprint(
        tabulate(GDF_Elv.loc[GDF_Elv['Diff'] < 0].head(), headers='keys', tablefmt='grid', showindex=False),
        verbose_in=True,
        verbose_out=verbose,
    )

    sprint('\tHorizontal segments:', indent=3, verbose_in=True, verbose_out=verbose)
    sprint(
        tabulate(GDF_Elv.loc[GDF_Elv['Diff'] == 0].head(), headers='keys', tablefmt='grid', showindex=False),
        verbose_in=True,
        verbose_out=verbose,
    )

# %% Check if problematic segments have multiple UStr segments
if debug_sfr:
    l_Incl = GDF_Elv.loc[GDF_Elv['Diff'] < 0, 'ID'].tolist()
    if len(l_Incl) > 0:
        l_Incl_multiple_UStr = []
        for S in l_Incl:
            sum = (GDF['DStr_ID'] == S).sum()
            if sum > 1:
                l_Incl_multiple_UStr.append((S, sum))
        if len(l_Incl_multiple_UStr) > 0:
            sprint(
                '\tSegments with Elv_UStr < Elv_DStr that have multiple UStr segments (complicates adjustment):',
                indent=3,
                verbose_in=True,
                verbose_out=verbose,
            )
            sprint(
                tabulate(
                    l_Incl_multiple_UStr,
                    headers=['ID', 'Number of UStr segments'],
                    tablefmt='grid',
                    showindex=False,
                ),
                verbose_in=True,
                verbose_out=verbose,
            )

# %%
"""
Elv correction algorithm:
To fix segments with Elv_UStr < Elv_DStr. Segments with Elv_UStr = Elv_DStr will be fixed by SFR itself (hopefully). The following abbreviations are useful for explaining the concept:
- A: DStr Elv of DStr segment
- B: UStr Elv of DStr segment
- C: DStr Elv of current segment
- D: UStr Elv of current segment
- F: DStr Elv of UStr segment(s)

Here is the idea behind the algorithm:
1. If C > D & B <= D :-> Set C = D
2. If C > D & B > D :-> Set C = D. Set B = D
3. If C <= D :-> No action.

Repeat till there are no segments with C < D.

When there is no downstream segment, we apply the logic used in case 1.

When there is only 1 UStr segment, the DStr Elv of the UStr segment can be modified to allow the UStr Elv of the current segmet to be increased as well, but if there are multiple, this becomes more complicated.
"""

sprint(
    'Applying elevation correction algorithm to segments where Elv_UStr < Elv_DStr.',
    verbose_in=True,
    verbose_out=verbose,
    indent=3,
    set_time=True,
    end='',
)
GDF_Elv = GDF_Elv.merge(
    GDF[['ID', 'Elv_UStr', 'Elv_DStr']], left_on='DStr_ID', right_on='ID', suffixes=('', '_DStrSeg'), how='left'
)
GDF_Elv[['A', 'B']] = GDF_Elv[['Elv_DStr_DStrSeg', 'Elv_UStr_DStrSeg']].copy()
GDF_Elv[['C', 'D']] = GDF_Elv[['Elv_DStr', 'Elv_UStr']].copy()
# GDF_Elv[GDF_Elv['B'].isna()]


def adjust_elevations(row):
    if row['C'] <= row['D']:  # If UStr Elv <= DStr Elv, no adjustment needed
        return row['B'], row['C']
    elif (row['C'] > row['D']) and (
        pd.isna(row['B'])
    ):  # If UStr Elv <= DStr Elv but DStr Elv is missing (OuFl segment)
        return pd.NA, row['D']
    elif (row['C'] > row['D']) and (row['B'] <= row['D']):
        return row['B'], row['D']
    elif (row['C'] > row['D']) and (row['B'] > row['D']):
        return row['D'], row['D']
    else:
        # Default case - should not happen, but ensures function always returns a tuple
        return row['B'], row['C']


GDF_Elv[['B_', 'C_']] = GDF_Elv.apply(adjust_elevations, axis=1, result_type='expand')
sprint('🟢', verbose_in=True, verbose_out=verbose, print_time=True)

# %% Print problematic segments after adjustment
# I'm worried consequtive segments might be problematic. Let's check if there are any.
if debug_sfr:
    GDF_Elv_unfixed = GDF_Elv[(GDF_Elv['Diff'] < 0)]
    consequtive = GDF_Elv_unfixed.loc[GDF_Elv_unfixed['DStr_ID'].isin(GDF_Elv_unfixed['ID']), 'DStr_ID']
    sprint(
        "Segments where Elv_UStr < Elv_DStr that are consequtive (i.e. one segment's DStr_ID is another segment's ID). Those might cause problems later.:",
        verbose_in=True,
        verbose_out=verbose,
        indent=3,
    )
    sprint(
        tabulate(
            GDF_Elv_unfixed.loc[
                (GDF_Elv_unfixed['ID'].isin(consequtive)) | (GDF_Elv_unfixed['DStr_ID'].isin(consequtive)),
                ['ID', 'DStr_ID', 'A', 'B', 'B_', 'C', 'C_', 'D'],
            ]
            .sort_values(by='D')
            .reset_index(drop=True),
            headers='keys',
            tablefmt='grid',
            showindex=False,
        ),
        verbose_in=True,
        verbose_out=verbose,
    )

    GDF_Elv['segment_drop'] = GDF_Elv['D'] - GDF_Elv['C_']
    GDF_Elv['DStr_drop'] = GDF_Elv['C_'] - GDF_Elv['B']
    sprint(
        'Segments where the adjusted DStr Elv (C_) is still smaller than the UStr Elv of the DStr segment (B). Those might cause problems later:',
        verbose_in=True,
        verbose_out=verbose,
        indent=3,
    )
    sprint(
        tabulate(
            GDF_Elv.loc[
                GDF_Elv['C_'] - GDF_Elv['B_'] < 0,
                ['ID', 'DStr_ID', 'A', 'B', 'B_', 'C', 'C_', 'D', 'segment_drop', 'DStr_drop'],
            ]
            .sort_values(by='DStr_drop')
            .reset_index(drop=True),
            headers='keys',
            tablefmt='grid',
            showindex=False,
        ),
        verbose_in=True,
        verbose_out=verbose,
    )

# %% Merge adjusted elevations back to GDF
GDF = GDF.merge(GDF_Elv[['ID', 'C_', 'D']], on='ID', how='left')

# %% Generate SFRmaker lines

GDF['width2'] = GDF['width'].copy()  # sfr.Lines.from_dataframe requires width 2 Col
lines = sfr.Lines.from_dataframe(
    df=GDF.copy(),  # .copy() to avoid GDF columns being renamed by function (this feels like a bug to me)
    id_column='ID',
    routing_column='DStr_ID',
    width1_column='width',
    width2_column='width2',
    dn_elevation_column='C_',
    up_elevation_column='D',
    name_column='CODE',
    width_units='m',
    height_units='m',
    crs=GDF.crs,
    #    shapefile=Pa_GPkg_1ry_SHP_SFR,
)

# %% Remove IDs that cause loops.
bad_ids = [i for i, p in lines.paths.items() if int(p[-1]) != 0]
lines.df = lines.df[~lines.df['id'].isin(bad_ids)].copy()


# %% [markdown]
