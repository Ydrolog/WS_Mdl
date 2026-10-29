# region 1. Init
# %% [markdown]
# ~~The aim of this notebook is to compare NBr60 against 52. In 54 qrun is much smaller for no apparent reason.~~
#
# ~~For this reason, I'll compare the MSW input files for both runs.~~
#
# ~~The comparrison outcome is recorded in ./compare_Ins-NBr60_Vs_NBr52.xlsx~~
#
# The aim of this notebook is to repeat the process, but for Sims NBr125 & NBr126. That's because we're increasing the Mdl area/boundaries, and need a new MSW_area_svat fix.
#
# Parts of the analysis are skipped cause we already understand the differences and just need the fix.

# %%

import pandas as pd
from WS_Mdl.core.mdl import Mdl_N

# %%
M = Mdl_N('NBr141')
MB = Mdl_N('NBr140', iMOD5=True)

Aa = M.Mdl_area - M.N_R * 2 * M.cellsize - (M.N_C - 2) * 2 * M.cellsize
Aa
# endregion

# region 3. Use NBr52's area in NBr61

# %% [markdown]
# # 3. Use NBr52's area in NBr61

# %%
Fi = 'idf_svat'
DF, DF_B = M.MSW_In.d[Fi].copy(), MB.MSW_In.d[Fi].copy()

# %%
Fi = 'area_svat'
DF_area, DF_B_area = M.MSW_In.d[Fi].copy(), MB.MSW_In.d[Fi].copy()

# %% Prep Cols for merge
DF.rename(
    columns={'row number of gridcell (MODFLOW style)': 'R', 'column number of gridcell (MODFLOW style)': 'C'},
    inplace=True,
)
DF_B.rename(
    columns={'row number of gridcell (MODFLOW style)': 'R', 'column number of gridcell (MODFLOW style)': 'C'},
    inplace=True,
)
DF_area.rename(
    columns={'row number of gridcell (MODFLOW style)': 'R', 'column number of gridcell (MODFLOW style)': 'C'},
    inplace=True,
)
DF_B_area.rename(
    columns={'row number of gridcell (MODFLOW style)': 'R', 'column number of gridcell (MODFLOW style)': 'C'},
    inplace=True,
)

DF['N'] = DF.groupby(['R', 'C']).cumcount()
DF_B['N'] = DF_B.groupby(['R', 'C']).cumcount()

# %%
DF = DF.merge(DF_area[['SVAT', 'area']], on='SVAT', how='left')
DF_B = DF_B.merge(DF_B_area[['SVAT', 'area']], on='SVAT', how='left')

# %%
DF_merge = DF.merge(DF_B, on=['R', 'C', 'N'], how='left', suffixes=('_S', '_B'))

# %%
DF_merge.drop(
    columns=[i for i in DF_merge.columns if i.startswith('X_') or i.startswith('Y_')] + ['SVAT_B'], inplace=True
)

# %%
DF_merge

# %%
DF_area['area'] = DF_merge['area_B'].fillna(DF_merge['area_S'])

# %%
import yaml

with open(r'g:\code\WS_Mdl\imod\msw\defaults\colspecs.yaml', 'r') as f:
    colspecs = yaml.safe_load(f)

# %%
DF_area

# %%
import numpy as np

col_widths = [j[1] - j[0] + 1 for j in colspecs['area_svat.inp']][: len(DF_area.columns)]

fmt_funcs = []
for i, c in enumerate(DF_area.columns):
    w = col_widths[i]
    if pd.api.types.is_float_dtype(DF_area[c]):
        max_val = DF_area[c].abs().max()
        pre_dec = int(np.floor(np.log10(max_val))) + 1 if max_val > 0 and not pd.isna(max_val) else 1
        post_dec = max(0, 5 - pre_dec)
        fmt_funcs.append(lambda val, w=w, dec=post_dec: f'{val:>{w}.{dec}f}' if not pd.isna(val) else ' ' * w)
    else:
        fmt_funcs.append(lambda val, w=w: f'{val:>{w}}' if not pd.isna(val) else ' ' * w)

Pa_Out = M.Pa.In / f'CAP/area_svat_patch/{M.MdlN}/area_svat.inp'
Pa_Out.parent.mkdir(parents=True, exist_ok=True)
with open(Pa_Out, 'w') as f:
    for row in DF_area.itertuples(index=False, name=None):
        formatted_row = ''.join(func(val) for func, val in zip(fmt_funcs, row)) + '\n'
        f.write(formatted_row)

# %%
DF

# %%
DF_area

# %%
