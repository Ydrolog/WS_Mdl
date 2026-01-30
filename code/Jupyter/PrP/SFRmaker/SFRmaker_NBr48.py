# %% [markdown]
# region Imports
# ## 0.0. Imports

# %%
import os
import re
import shutil as sh
from datetime import datetime as DT
from os.path import basename as PBN
from os.path import dirname as PDN
from os.path import join as PJ
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import primod
import sfrmaker as sfr
import WS_Mdl.geo as G
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM
from imod import mf6, msw
from imod.mf6 import ConstantHead
from scipy.spatial.distance import cdist
from shapely.geometry import box

# %%
# import importlib as IL
# IL.reload(U)
# IL.reload(UIM)
# IL.reload(G)
print('🟢 - Imports successful!')
# endregion

# %% [markdown]
# region Options
# ## 0.1. Options

# %%
MdlN = 'NBr48'
MdlN_SFR_In = 'NBr40'
Pa_Gpkg_ = PJ(U.Pa_WS, rf'g:\models\NBr\PrP\SFR\BrabantseDelta\Gpkg\WBD_detail_SW_NW_cleaned_{MdlN_SFR_In}.gpkg')

Pa_HD_OBS = r'g:\models\NBr\In\OBS\HD\NBr34\NBr34.HD.OBS6'
Pa_RIV_OBS = r'g:\models\NBr\In\OBS\RIV\NBr45\NBr45.RIV.OBS6'

U.set_verbose(False)

d_Pa = U.get_MdlN_Pa(MdlN)
Pa_PRJ = d_Pa['PRJ']
Dir_PRJ = PDN(Pa_PRJ)
d_INI = U.INI_to_d(d_Pa['INI'])
Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = U.Mdl_Dmns_from_INI(d_Pa['INI'])
SP_date_1st, SP_date_last = [DT.strftime(DT.strptime(d_INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']]
dx = dy = float(d_INI['CELLSIZE'])

# %%
l_X_Y_Cols = ['Xstart', 'Ystart', 'Xend', 'Yend']
l_Circ_IDs = [6561, 8788, 18348]
print('🟢 - Options set successfully!')
# endregion

# %% [markdown]
# region Load Model
# # 1. Load Model Ins

# %% [markdown]
# ## 1.0. Load PRJ

# %%
PRJ_, PRJ_OBS = UIM.o_PRJ_with_OBS(Pa_PRJ)
PRJ, period_data = PRJ_[0], PRJ_[1]
print('🟢 - PRJ loaded successfully!')

# %% [markdown]
# ## 1.1. Load DIS and limit to Mdl Aa

# %%
PRJ_regrid = UIM.regrid_PRJ(PRJ, MdlN)
BND = PRJ_regrid['bnd']['ibound']

# %%
# Set outer boundaries to -1 (for CHD)

# Get the coordinate indices for boundaries
y_coords = BND.y
x_coords = BND.x
first_y = y_coords.isel(y=0)  # First y coordinate
last_y = y_coords.isel(y=-1)  # Last y coordinate
first_x = x_coords.isel(x=0)  # First x coordinate
last_x = x_coords.isel(x=-1)  # Last x coordinate

# Set boundary values using .loc indexing
BND.loc[:, first_y, :] = -1  # Top row (all layers, first y, all x)
BND.loc[:, last_y, :] = -1  # Bottom row (all layers, last y, all x)
BND.loc[:, :, first_x] = -1  # Left column (all layers, all y, first x)
BND.loc[:, :, last_x] = -1  # Right column (all layers, all y, last x)

# %%
BND.isel(layer=0, x=range(0, 10), y=range(0, 10)).plot.imshow(cmap='viridis')
print('🟢 - Boundaries set successfully!')

# %% [markdown]
# ## 1.2. Load MF6 Mdl

# %%
times = pd.date_range(SP_date_1st, SP_date_last, freq='D')

# %%
Sim_MF6 = mf6.Modflow6Simulation.from_imod5_data(PRJ_regrid, period_data, times)
print('🟢 - MF6 model loaded successfully!')

# %%
MF6_Mdl = Sim_MF6['imported_model']

# %%
MF6_Mdl['oc'] = mf6.OutputControl(save_head='last', save_budget='last')
Sim_MF6['ims'] = UIM.mf6_solution_moderate_settings()  # Mimic iMOD5's "Moderate" settings

# %%
MF6_DIS = MF6_Mdl['dis']  # This gets the OLD 100m grid

# %% [markdown]
# ## 1.3. Load MSW

# %% [markdown]
# ### 1.3.0. Fix mete_grid.inp relative paths

# %%
# Replace the mete_grid.inp path in the PRJ_MSW_for_MSW dictionary
PRJ['extra']['paths'][2][0] = UIM.mete_grid_Cvt_to_AbsPa(Pa_PRJ, PRJ_regrid)

# %% [markdown]
# ### 1.3.2. Finally load MSW Sim

# %%
# Create the MetaSwap model
PRJ_MSW = {'cap': PRJ_regrid.copy()['cap'], 'extra': PRJ_regrid.copy()['extra']}
MSW_Mdl = msw.MetaSwapModel.from_imod5_data(PRJ_MSW, MF6_DIS, times)
print('🟢 - MetaSwap model loaded successfully!')

# %% [markdown]
# ## 1.4. Connect MF6 to MetaSWAP

# %% [markdown]
# ### 1.4.1. Clip models

# %%
Sim_MF6_AoI = Sim_MF6.clip_box(x_min=Xmin, x_max=Xmax, y_min=Ymin, y_max=Ymax)
MF6_Mdl_AoI = Sim_MF6_AoI['imported_model']

# %%
MSW_Mdl_AoI = MSW_Mdl.clip_box(x_min=Xmin, x_max=Xmax, y_min=Ymin, y_max=Ymax)

# %%
print(f'MF6 Model AoI DIS shape: {MF6_Mdl_AoI["dis"].dataset.sizes}')
print(f'MSW Model AoI grid shape: {MSW_Mdl_AoI["grid"].dataset.sizes}')
print('🟢 - Both models successfully clipped to Area of Interest with compatible discretization!')

# %% [markdown]
# ## 1.5. Load & Cleanup models

# %% [markdown]
# ### 1.5.0. Load

# %%
for pkg in MF6_Mdl_AoI.values():
    pkg.dataset.load()

for pkg in MSW_Mdl_AoI.values():
    pkg.dataset.load()

# %% [markdown]
# ### 1.5.1. MF6 mask

# %%
# Create mask from current regridded model (not the old one)
mask = MF6_Mdl_AoI.domain

# %%
# Fix CHD package layer ordering issue (layers must be monotonically increasing)

chd_pkg = Sim_MF6_AoI['imported_model']['chd_merged']
head_data_sorted = chd_pkg.dataset['head'].load().sortby('layer')
Sim_MF6_AoI['imported_model']['chd_merged'] = ConstantHead(head=head_data_sorted, validate=False)

# %%
Sim_MF6_AoI.mask_all_models(mask)
DIS_AoI = MF6_Mdl_AoI['dis']

# %% [markdown]
# ### 1.5.2. Cleanup MF6

# %%
try:
    for Pkg in [i for i in MF6_Mdl_AoI.keys() if ('riv' in i.lower()) or ('drn' in i.lower())]:
        MF6_Mdl_AoI[Pkg].cleanup(DIS_AoI)
except:
    print('Failed to cleanup packages. Proceeding without cleanup. Fingers crossed!')

# %% [markdown]
# ### 1.5.3 Cleanup MetaSWAP

# %%
MSW_Mdl_AoI['grid'].dataset['rootzone_depth'] = MSW_Mdl_AoI['grid'].dataset['rootzone_depth'].fillna(1.0)

# %% [markdown]
# ## 1.6. Couple & Write

# %%
metamod_coupling = primod.MetaModDriverCoupling(
    mf6_model='imported_model', mf6_recharge_package='msw-rch', mf6_wel_package='msw-sprinkling'
)
metamod = primod.MetaMod(MSW_Mdl_AoI, Sim_MF6_AoI, coupling_list=[metamod_coupling])

# %%
os.makedirs(d_Pa['Pa_MdlN'], exist_ok=True)  # Create simulation directory if it doesn't exist

# %%
# Use correct paths from d_Pa instead of hardcoded paths
Pa_MF6_DLL = d_Pa['MF6_DLL']
Pa_MSW_DLL = d_Pa['MSW_DLL']
Pa_IMC = d_Pa['coupler_Exe']

# %%
metamod.write(
    directory=d_Pa['Pa_MdlN'], modflow6_dll=Pa_MF6_DLL, metaswap_dll=Pa_MSW_DLL, metaswap_dll_dependency=PDN(Pa_MF6_DLL)
)
print('🟢 - Coupled model written successfully!')
# endregion

# %% [markdown]
# region SFRlines
# # 2. Create SFR lines

# %% [markdown]
# ## 2.1.Load

# %%
Pa_Gpkg = PJ(U.Pa_WS, r'g:\models\NBr\PrP\SFR\BrabantseDelta\Gpkg\WBD_detail_SW_NW_cleaned.gpkg')

# %%
GDF = gpd.read_file(Pa_Gpkg)

GDF0 = GDF.copy()

# %%
GDF = U.GDF_clip_Mdl_Aa(GDF, d_Pa['INI'])

# %%
GDF1 = GDF.copy()

# %%
GDF1.shape

# %%
GDF1.describe(include='all')

# %% [markdown]
# ## 2.2 Ensure slope

# %% [markdown]
# #### Upstream and downstream elevations

# %%
GDF[['ID', 'Elv_UStr', 'Elv_DStr']].describe(include='all')

# %% [markdown]
# 🟢 - No nulls + the percentiles make sense.<br>
# Let's make sure the UStr is always higher than the DnStr.<br>
# Then let's print out some values to check in QGIS.

# %%
(
    (GDF['Elv_UStr'] <= GDF['Elv_DStr']).sum(),
    (GDF['Elv_UStr'] < GDF['Elv_DStr']).sum(),
    (GDF['Elv_UStr'] > GDF['Elv_DStr']).sum(),
    GDF.shape[0],
)

# %% [markdown]
# We will assume SFRmaker will work where Elv_UStr >= Elv_DStr, so we'll only adjust those where Elv_UStr < Elv_DStr.

# %% [markdown]
# #### Let's print out some CODEs where =, to check in QGIS. *(We don't really need to, I'm just curious)*

# %%
GDF_Elv = GDF[['ID', 'Elv_UStr', 'Elv_DStr', 'DStr_code', 'DStr_ID']].copy()

# %%
GDF_Elv['Diff'] = GDF_Elv['Elv_UStr'] - GDF_Elv['Elv_DStr']

# %%
GDF_Elv.loc[GDF_Elv['Diff'] == 0].head()

# %%
GDF_Elv.loc[GDF_Elv['Diff'] < 0].sort_values(by='Diff', ascending=True).head()

# %% [markdown]
# ##### Let's see if any of the problematic segments have multiple UStr segments. That would make a solution harder to implement.<br>
# *(if there is only 1 UStr segment, the DStr Elv of the UStr segment can be modified to allow the UStr Elv of the current segmet to be increased as well, but if there are multiple, this becomes more complicated)*

# %%
l_problematic = GDF_Elv.loc[GDF_Elv['Diff'] < 0, 'ID'].tolist()
for S in l_problematic:
    sum = (GDF['DStr_ID'] == S).sum()
    if sum > 1:
        print(S, sum)

# %% [markdown]
# ##### Elv correction algorithm

# %% [markdown]
# We'll design an algorithm to fix those with <. Those with = will be fixed by SFR itself (hopefully). The following abbreviations are useful for explaining the concept:
# - A: DStr Elv of DStr segment
# - B: UStr Elv of DStr segment
# - C: DStr Elv of current segment
# - D: UStr Elv of current segment
# - F: DStr Elv of UStr segment(s)
#
# Here is the idea behind the algorithm:
# 1. If **C > D & B <= D** :<br>
# -> Set **C = D**
# 2. If **C > D & B > D** :<br>
# -> Set **C = D**. Set **B = D**
# 3. If **C <= D** :<br>
# -> **No action**.
#
# Repeat till there are no segments with C < D.
#
# When there is no downstream segment, we apply the logic used in case 1.

# %%
GDF_Elv = GDF_Elv.merge(
    GDF[['ID', 'Elv_UStr', 'Elv_DStr']], left_on='DStr_ID', right_on='ID', suffixes=('', '_DStr'), how='left'
)

# %%
GDF_Elv[['A', 'B']] = GDF_Elv[['Elv_UStr_DStr', 'Elv_DStr_DStr']].copy()

# %%
GDF_Elv[['C', 'D']] = GDF_Elv[['Elv_UStr', 'Elv_DStr']].copy()

# %%
GDF_Elv[GDF_Elv['B'].isna()]


# %%
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


# %%
GDF_Elv[['B_', 'C_']] = GDF_Elv.apply(adjust_elevations, axis=1, result_type='expand')

# %% [markdown]
# I'm worried consequtive segments might be problematic. Let's check if there are any.

# %%
GDF_Elv_unfixed = GDF_Elv[(GDF_Elv['Diff'] < 0)]
consequtive = GDF_Elv_unfixed.loc[GDF_Elv_unfixed['DStr_ID'].isin(GDF_Elv_unfixed['ID']), 'DStr_ID']
GDF_Elv_unfixed.loc[
    (GDF_Elv_unfixed['ID'].isin(consequtive)) | (GDF_Elv_unfixed['DStr_ID'].isin(consequtive)),
    ['ID', 'DStr_ID', 'A', 'B', 'B_', 'C', 'C_', 'D'],
].sort_values(by='D').reset_index(drop=True)

# %% [markdown]
# Consequtive not ok. Let's hope that SFRmaker can handle this. Otherwise we'll have to come back later.

# %%
GDF_Elv.loc[GDF_Elv['D'] - GDF_Elv['C_'] < 0]

# %% [markdown]
# Cool, no segments without any drop in Elv.

# %%
GDF_Elv['segment_drop'] = GDF_Elv['D'] - GDF_Elv['C_']
GDF_Elv['DStr_drop'] = GDF_Elv['C_'] - GDF_Elv['B']
GDF_Elv.loc[
    GDF_Elv['C_'] - GDF_Elv['B_'] < 0, ['ID', 'DStr_ID', 'A', 'B', 'B_', 'C', 'C_', 'D', 'segment_drop', 'DStr_drop']
].sort_values(by='DStr_drop').reset_index(drop=True)

# %% [markdown]
# There are **quite a few** segments where C_ > B!!! SFRmaker might fix this. If not, I'll come back and fix it.

# %%
GDF2 = GDF.copy()

# %%
GDF = GDF.merge(GDF_Elv[['ID', 'C_', 'D']], on='ID', how='left')

# %% [markdown]
# ## 2.3 Remove DStr_IDs that are outside the model

# %%
GDF3 = GDF.copy()

# %%
GDF_DStr_Out_Mdl_Aa = GDF.loc[~GDF['DStr_ID'].isin(GDF['ID']) & GDF['DStr_ID'] != 0]

# %%
GDF.loc[~GDF['DStr_ID'].isin(GDF['ID']) & GDF['DStr_ID'] != 0, 'DStr_ID'] = 0

# %%
len(GDF.loc[~GDF['DStr_ID'].isin(GDF['ID']) & GDF['DStr_ID'] != 0, 'DStr_ID'])

# %% [markdown]
# ## 2.4 Remove circular IDs

# %%
GDF = GDF.loc[~GDF['DStr_ID'].isin(l_Circ_IDs)]

# %% [markdown]
# ## 2.5 Generate SFRmaker lines

# %%
GDF['width2'] = GDF['width'].copy()

# %%
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

# %%
DF_lines = lines.df
U.DF_info(lines.df)
print('🟢 - SFR lines generated successfully!')
# endregion

# %% [markdown]
# region Connect SFR to MF6
# # 3. Connect SFR to MF6 model

# %% [markdown]
# ## 3.0. Create SFR_grid item

# %% [markdown]
# ### 3.0.0 Initiate parameters

# %%
# Create sfr.StructuredGrid directly from MF6_DIS (DataFrame approach) #666 This cell and the cells below it can be combined into a function to read in a MF6_DIS (imod) object, and return a DF (GDF_grid) with the grid and geometry.
DS = MF6_DIS.dataset
N_L, N_R, N_C = DS.dims['layer'], DS.dims['y'], DS.dims['x']
dx, dy = abs(float(DS.coords['dx'].values)), abs(float(DS.coords['dy'].values))
Ls, Xs, Ys = DS.coords['layer'].values, DS.coords['x'].values, DS.coords['y'].values
X_Ogn, Y_Ogn = Xs[0] - dx / 2, Ys[0] + dy / 2  # Upper-left corner

# %%
# Construct TOP, BOT. TOP array: 1st layer from DS['top'], rest from DS['bottom'][::-1] with layer+1
TOPs = np.zeros((N_L, N_R, N_C))
TOPs[0] = DS['top'].values
TOPs[1:] = DS['bottom'].sel(layer=range(1, N_L))
BOTs = DS['bottom'].values  # Shape: (N_L, N_R, N_C)

# %%
# Create full 3D grid indices
k, i, j = np.meshgrid(range(N_L), range(N_R), range(N_C), indexing='ij')
k, i, j = k.ravel(), i.ravel(), j.ravel()

# %% [markdown]
# ### 3.0.1 Prepare GDF

# %%
GDF_grid = gpd.GeoDataFrame(
    {
        'k': k,
        'i': i,
        'j': j,
        'node': range(N_L * N_R * N_C),
        'isfr': 1,  # All cells can potentially have SFR # if function is made out of this, this needs to be removed and added to the DF after the function has run.
        'top': TOPs.ravel(),
        'bottom': BOTs.ravel(),
    }
)

# %%
mask = GDF_grid['k'].eq(0)
i_L0 = GDF_grid.loc[mask, 'i'].to_numpy()
j_L0 = GDF_grid.loc[mask, 'j'].to_numpy()

# %%
xmin = X_Ogn + j_L0 * dx
xmax = X_Ogn + (j_L0 + 1) * dx
ymin = Y_Ogn - (i_L0 + 1) * dy
ymax = Y_Ogn - i_L0 * dy

# %%
L0_geom = [box(x0, y0, x1, y1) for x0, y0, x1, y1 in zip(xmin, ymin, xmax, ymax)]

# %%
for k in GDF_grid['k'].unique():
    GDF_grid.loc[GDF_grid['k'] == k, 'geometry'] = L0_geom

# %%
GDF_grid = GDF_grid.set_geometry('geometry', crs=DS.rio.crs)

# %% [markdown]
# ### 3.0.2 Identify deepest SFR layer

# %% [markdown]
# The reason we're doing this is that the model has too many Ls and it takes a very long time to run the SFR functions with all of them. So we'll find the deepest L that has any part of the stream network in it, and **we'll only use up to that layer for the SFR grid.**

# %%
for L in range(BOTs.shape[0]):
    L_BOT_min = BOTs[L].min()
    L_BOT_max = BOTs[L].max()
    print(L + 1, f'|{L_BOT_min:8.2f} |', f'{L_BOT_max:8.2f} |')
    if L_BOT_min > DF_lines['elevdn'].min():
        SFR_deepest_L = L + 1

# %% [markdown]
# ### 3.0.3 Create SFR grid(s)

# %%
SFR_grid = sfr.StructuredGrid(
    GDF_grid.loc[GDF_grid['k'] <= SFR_deepest_L - 1], crs=G.crs
)  # -1 cause grid k starts at 0, L at 1

# %%
SFR_grid_L1 = sfr.StructuredGrid(GDF_grid.loc[GDF_grid['k'] == 0], crs=G.crs)  # Extract layer 1 (k=0)

# %%
# Check what type of object and its basic info without triggering full repr
print(f'SFR_grid object created: {SFR_grid is not None}')

# Check if it has expensive methods for representation
# print(f'Available methods: {[method for method in dir(SFR_grid) if not method.startswith("_")][:10]}')

# # Try to get basic info without full representation
# try:
#     print(f'Grid shape info: {hasattr(SFR_grid, "shape")}')
#     if hasattr(SFR_grid, 'nlay'):
#         print(f'Number of layers: {SFR_grid.nlay}')
#     if hasattr(SFR_grid, 'nrow'):
#         print(f'Number of rows: {SFR_grid.nrow}')
#     if hasattr(SFR_grid, 'ncol'):
#         print(f'Number of cols: {SFR_grid.ncol}')
# except Exception as e:
#     print(f'Error getting basic info: {e}')

# %% [markdown]
# ## 3.2. SFRdata

# %% [markdown]
# ### 3.2.0 Init

# %%
GDF4 = GDF.copy()

# %%
GDF = GDF4.copy()

# %%
paths = lines.paths

# %%
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

# %%
bad_ids = [i for i, p in lines.paths.items() if int(p[-1]) != 0]
lines.df = lines.df[~lines.df['id'].isin(bad_ids)].copy()

# %%
SFR_data = lines.to_sfr(grid=SFR_grid_L1, one_reach_per_cell=True)

# %% [markdown]
# ### 3.2.1 Explore DF_reach

# %%
SFR_data.reach_data.sort_values(by=['i', 'j'])

# %%
DF_reach = SFR_data.reach_data.copy()
DF_reach[['k', 'i', 'j']] = DF_reach[['k', 'i', 'j']] + 1  # convert to 1-based indexing for reviewing

# %%
DF_reach.describe()  # include='all')

# %% [markdown]
# Some comments regarding DF_reaches: #666 Needs to be re-done
# - We have a large **number of reaches** (rno.max()=7819), and all columns have the same number of valid values, which is good.
# - **k** wasn't filled properly. We need to use the assign_layer function to fix this. **Surprise...<br>There are 2...<br>
# <t> sfrmaker.sfrdata.assign_layers <br>
# <t> sfrmaker.utils.assign_layers <br>
# We'll use the latter, where we can use BOTs. The other one requires a full loaded flopy model. <t>**
# - **j** is within range, so it was probably calculated correctly.
# - **iseg** makes sense. **ireach** is the reach number within the segment (according to copilot), seems feasible.
# - **width** has a few values that are too big. Let's print them out to check in QGIS.
# - **rchlen, slope, strtop** all make sense.
# - **strthick** is 1 everywhere. We need to edit this, based on some sort of assumption and the conductance value of the equivalent RIV item. Let's start with strthick=0.1 (cause 1m is too much).
# - **strhc1** is set to 0.1 m/d, as default value of 1 seems too high.
# - **thts**, **thti**, **eps** & **uhc** are not used as far as I know.
# - **outreach** seems iffy, as it's float, while I was expecting an int.
# - how can **asum** be negative?

# %%
DF_reach['strthick'] = 0.1  # Set a default streambed thickness of 0.1 m
DF_reach['strhc1'] = 0.1  # Set a default streambed hydraulic conductivity of 0.1 m/d

# %% [markdown]
# #### Explore width

# %%
DF_reach.loc[
    :,
    [
        'rno',
        'outreach',
        'iseg',
        'outseg',
        'node',
        'k',
        'i',
        'j',
        'name',
        'rchlen',
        'width',
        'strtop',
        'strthick',
        'asum',
    ],
].sort_values(by=['width', 'i', 'j'], ascending=[False, True, True])

# %% [markdown]
# I'll set all widths > 100 m to 1 m for now. #666

# %%
DF_reach.loc[DF_reach['width'] > 100, 'width'] = 1

# %% [markdown]
# ### 3.2.2 Assign the correct layers - k.

# %%
DF_reach[['k', 'i', 'j']] = (
    DF_reach[['k', 'i', 'j']] - 1
)  # convert to 0-based indexing for utils_assign_layers function

# %%
reach_Ls, strtps = sfr.utils.assign_layers(reach_data=DF_reach, botm_array=BOTs, pad=0)

# %%
DF_reach['k'] = reach_Ls

# %% [markdown]
# ### 3.2.3 Check
# Examples to check if segments were connected to the right cells

# %%
for i, seg in enumerate(DF_reach['name'].unique()[:10]):
    print(i + 1, seg, DF_reach.loc[DF_reach['name'] == seg, 'name'].count())

# %%
DF_reach[['k', 'i', 'j']] = DF_reach[['k', 'i', 'j']] + 1  # convert to 1-based indexing for reviewing

# %%
DF_reach.loc[
    DF_reach['name'] == 'OVK01451',
    [
        'rno',
        'outreach',
        'iseg',
        'outseg',
        'node',
        'k',
        'i',
        'j',
        'name',
        'rchlen',
        'width',
        'strtop',
        'strthick',
        'asum',
    ],
].sort_values(by=['i', 'j'])

# %%
DF_reach.loc[
    DF_reach['name'] == 'OVK02048',
    [
        'rno',
        'outreach',
        'iseg',
        'outseg',
        'node',
        'k',
        'i',
        'j',
        'name',
        'rchlen',
        'width',
        'strtop',
        'strthick',
        'asum',
    ],
].sort_values(by=['name', 'j', 'i'])

# %%
DF_reach.loc[
    DF_reach['name'] == 'OVK20466',
    [
        'rno',
        'outreach',
        'iseg',
        'outseg',
        'node',
        'k',
        'i',
        'j',
        'name',
        'rchlen',
        'width',
        'strtop',
        'strthick',
        'asum',
    ],
].sort_values(by=['name', 'j', 'i'])

# %%
DF_reach[['k', 'i', 'j']] = DF_reach[['k', 'i', 'j']] - 1  # convert to 0-based indexing for SFRmaker operations

# %% [markdown]
# ### 3.2.4 Apply RIV conductance to DF_reach

# %% [markdown]
# ##### Calculate Default Conductance

# %%
DF_RC = DF_reach.copy()[
    ['rno', 'name', 'k', 'i', 'j', 'iseg', 'outseg', 'rchlen', 'width', 'strtop', 'strthick', 'strhc1', 'asum']
]

# %%
DF_RC['Cond'] = DF_RC['width'] * DF_RC['rchlen'] * DF_RC['strhc1'] / DF_RC['strthick']

# %% [markdown]
# ##### Import RIV Cond shapefile.

# %% [markdown]
# - Contrary to the previous Sim there will be no averaging of conductances cause Detailwatergangen is the only effective one. (For more info, check the notes presentations or QGIS)
# - Detailwatergangen conductance is missing in some places though, so we'll still load RIV_Drn and use it's conductance there.

# %%
Pa_Cond_A = PJ(U.Pa_WS, r'models\NBr\In\RIV\RIV_Cond_DETAILWATERGANGEN_NBr1.IDF')
Pa_Cond_B = PJ(U.Pa_WS, r'models\NBr\In\RIV\RIV_Cond_DRN_NBr1.IDF')

# %%
A = imod.idf.open(Pa_Cond_A).sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
B = imod.idf.open(Pa_Cond_B).sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))

# %%
print(
    f'A values >0: {(A > 1).sum().compute().values.item()} / {A.size} ({(A > 1).sum().compute().values.item() / A.size:.2%}),\nB values >0: {(B > 1).sum().compute().values.item()} / {B.size} ({(B > 1).sum().compute().values.item() / B.size:.2%})'
)

# %%
C = A.where(A > 0, B)

# %%
DF_RC['RIV_Cond'] = DF_RC[
    'Cond'
].copy()  # Apply conductance matching to DF_RC using array A. Start with copy of existing Cond values as fallback

C_DF_RC = C.values[
    DF_RC['i'].values, DF_RC['j'].values
]  # Get array values for all i,j coordinates at once (vectorized)

# %%
# Replace only where array has valid (non-NaN) values
valid_mask_RC = ~np.isnan(C_DF_RC)
DF_RC.loc[valid_mask_RC, 'RIV_Cond'] = C_DF_RC[valid_mask_RC]

# %%
print('DF_RC conductance matching results:')
print(
    f'Replaced {valid_mask_RC.sum()} values out of {len(DF_RC)} total rows ({valid_mask_RC.sum() / len(DF_RC) * 100:.1f}%)'
)
print(f'Original Cond: min={DF_RC["Cond"].min():.3f}, max={DF_RC["Cond"].max():.3f}')
print(f'New RIV_Cond: min={DF_RC["RIV_Cond"].min():.3f}, max={DF_RC["RIV_Cond"].max():.3f}')

# Check how many values actually changed
changed_values_RC = DF_RC['Cond'] != DF_RC['RIV_Cond']
print(f'Values that changed: {changed_values_RC.sum()} out of {len(DF_RC)}')

# %%
DF_RC['K_RIV'] = DF_RC['RIV_Cond'] * DF_RC['strthick'] / (DF_RC['width'] * DF_RC['rchlen'])

# %%
DF_RC['Cond_Diff'] = DF_RC['RIV_Cond'] - DF_RC['Cond']

# %%
DF_reach['strhc1'] = DF_RC['K_RIV']  # Set it back to DF_reach

# %% [markdown]
# ### 3.2.5 Explore segments

# %%
DF_Sgm = SFR_data.segment_data.copy()

# %%
DF_Sgm.iloc[:].describe()

# %% [markdown]
# Most columns aren't interesting. Let's plot the interesting ones.

# %%
DF_Sgm[
    [
        'nseg',
        'outseg',
        'roughch',
        'elevup',
        'elevdn',
        'width1',
        'width2',
    ]
]

# %%
(DF_Sgm['width1'] == DF_Sgm['width1']).all()

# %%
(DF_Sgm['elevup'] >= DF_Sgm['elevdn']).all()

# %% [markdown]
# We can see:
# - the roughness values are all the same (default) - **OK**
# - downstream elevation is always lower than (or equal to) upstream - **OK**
# - the widths seem to be the ones read from the shapefile - **OK**

# %% [markdown]
# ### 3.2.6 Add SFR OBS

# %% [markdown]
# #### Calibration points

# %%
Pa_SFR_OBS_In = PJ(
    d_Pa['In'], 'OBS/SFR/NBr40/NBr40_SFR_OBS_Pnt.csv'
)  # 666 Should be PJ(d_Pa['In'], f'OBS/SFR/{MdlN}/{MdlN}_SFR_OBS_Pnt.csv')
DF_SFR_OBS = pd.read_csv(Pa_SFR_OBS_In)
DF_SFR_OBS

# %%
for (
    i,
    row,
) in DF_SFR_OBS.iterrows():  # Have to add them one by one, otherwise it groups them by reach and only keeps the 1st one. This is an SFRmaker bug, I can fix that later and make a pull request. #666 it worked for stage though, so maybe I should trty again.
    SFR_data.add_observations(
        pd.DataFrame(row).T,
        x_location_column='x',
        y_location_column='y',
        obstype_column='obstype',
        obsname_column='site_no',
    )

# %% [markdown]
# #### Stage

# %%
DF_stage_OBS = pd.DataFrame({'rno': DF_reach['rno']})

# %%
DF_stage_OBS['obs_name'] = (
    'Stg_L'
    + (DF_reach['k'] + 1).astype(str)
    + '_R'
    + (DF_reach['i'] + 1).astype(str)
    + '_C'
    + (DF_reach['j'] + 1).astype(str)
)
DF_stage_OBS['obstype'] = 'stage'

# %%
DF_reach.shape

# %%
SFR_data.add_observations(DF_stage_OBS, rno_column='rno', obstype_column='obstype', obsname_column='obs_name')

# %%
SFR_data.observations

# %% [markdown]
# ### 3.2.7 Run diagnostics

# %%
SFR_data.run_diagnostics(verbose=True)

# %% [markdown]
# Most checks passed, except for: #666 need to re-check
# 1. Checking reach_data for downstream rises in streambed elevation...<br>68 reaches encountered with strtop < strtop of downstream reach. Let's see if this causes a problem.
# 2. Checking for model cells with multiple non-zero SFR conductances...
# 565 model cells with multiple non-zero SFR conductances found.
# This can be fixed easily with one of the SFRdata options. We'll come here if it causes an error in the Sim.
# 3. floppy Mdl not connected to SFRdata means:<br>
#     3.1 Cannot check reach proximities
#     3.2 Cannot check streambed elevations against cell bottom elevations. This shouldn't be a problem as the assign_layers function uses strbedthck (to assign k).
#

# %%
GDF_Elv.loc[GDF_Elv['D'] - GDF_Elv['B_'] < 0]

# %% [markdown]
# There are fewer entries in the GDF_Elv where the DStr Elv > UStr Elv, but this DF contains segments, not reaches. So this is expected.

# %% [markdown]
# ## 3.3 Write file and add to NAM

# %%
SFR_data.reach_data = DF_reach

# %%
SFR_data.write_package(d_Pa['SFR'], version='mf6')

# %%
# Try to find an inteernal SFRmaker way to fix this later. This is just a temporary patch.
with open(d_Pa['SFR'], 'r+', encoding='cp1252') as f:
    content = f.read()
    content = content.replace(f'FILEIN {MdlN}.SFR6.obs', f'FILEIN imported_model/{MdlN}.SFR6.obs')
    content = content.replace('BUDGET FILEOUT', '#BUDGET FILEOUT')
    f.seek(0)
    f.truncate()
    f.write(content)

# %%
sh.copy2('model_SFR.chk', PJ(d_Pa['MF6'], 'imported_model/model_SFR.chk'))
os.remove('model_SFR.chk')

# %%
with open(d_Pa['NAM_Mdl'], 'r') as f1:
    l_Lns_NAM = f1.readlines()

# %%
l_Lns_NAM.insert(-1, f'  sfr6 imported_model/{PBN(d_Pa["SFR"])} sfr\n')

# %%
with open(d_Pa['NAM_Mdl'], 'w') as f2:
    f2.writelines(l_Lns_NAM)
print('🟢 - SFR package added to NAM file.')
# endregion

# %% [markdown]
# region Connect DRN to SFR
# # 4. Connect DRN to SFR

# %% [markdown]
# ### 4.3.1 Prepare DF

# %%
base = PJ(d_Pa['Pa_MdlN'], 'modflow6/imported_model')
folders = [f for f in os.listdir(base) if ('drn' in f.lower()) and '.' not in f and os.path.isdir(PJ(base, f))]
l_DRN_Pa = [
    PJ(base, folder, fname)
    for folder in folders
    for fname in os.listdir(PJ(base, folder))
    if os.path.isfile(PJ(base, folder, fname))
]
# l_DRN_Pa  # list of full paths to files inside the matched "drn" folders


# %%
def read_mf6_drn_bin(filepath: str | Path) -> pd.DataFrame:
    """Read MODFLOW 6 DRN binary input (imod format) into a DataFrame."""
    dtype = np.dtype(
        [
            ('k', '<i4'),  # layer
            ('i', '<i4'),  # row
            ('j', '<i4'),  # column
            ('elev', '<f8'),  # elevation
            ('cond', '<f8'),  # conductance
        ]
    )
    path = Path(filepath)
    nrec = path.stat().st_size // dtype.itemsize
    arr = np.fromfile(path, dtype=dtype, count=nrec)
    return pd.DataFrame(arr)


# %%
d_DRN_DF = {}

for i in range(len(l_DRN_Pa)):
    DF_DRN = read_mf6_drn_bin(l_DRN_Pa[i])
    d_DRN_DF[int(re.search(r'(?i)drn[-_]?(\d+)', PDN(l_DRN_Pa[i])).group(1))] = DF_DRN.loc[
        ~DF_DRN['i'].isin([1, N_R]) & ~DF_DRN['j'].isin([1, N_C])
    ]

# %%
for k in d_DRN_DF.keys():
    # print(f"DRN-{k} DataFrame shape: {d_DRN_DF[k].shape}")
    d_DRN_DF[k] = U.Calc_DF_XY(d_DRN_DF[k], X_Ogn, Y_Ogn, cellsize)
    d_DRN_DF[k].drop(columns=['cond', 'elev'], inplace=True)
    d_DRN_DF[k]['Pkg1'] = f'drn-{k}'
    d_DRN_DF[k]['Pvd_ID'] = d_DRN_DF[k].index + 1  # 1-based index

# %%
DF_reach_for_DRN = U.Calc_DF_XY(DF_reach[['rno', 'i', 'j']], X_Ogn, Y_Ogn, cellsize)

# %%
# Combine all DRN DataFrames and match with reach points by minimum distance

# Combine all d_DRN_DF items into a single DataFrame
DF_DRN_all = pd.concat(d_DRN_DF.values(), ignore_index=True)

# Calculate distances and find closest reach for each DRN point
drn_coords = DF_DRN_all[['X', 'Y']].values
reach_coords = DF_reach_for_DRN[['X', 'Y']].values
distances = cdist(drn_coords, reach_coords, metric='euclidean')
min_indices = np.argmin(distances, axis=1)

# Add matched reach data to DRN DataFrame
matched_reach_data = DF_reach_for_DRN.iloc[min_indices].reset_index(drop=True)
DF_DRN_all_matched = DF_DRN_all.copy()
DF_DRN_all_matched['Rcv_ID'] = matched_reach_data['rno'].values
DF_DRN_all_matched['distance_to_match'] = distances[np.arange(len(drn_coords)), min_indices]

print(f'Combined {len(DF_DRN_all):,} DRN points from {len(d_DRN_DF)} DataFrames')
print(f'Matched to {DF_DRN_all_matched["Rcv_ID"].nunique()} unique reaches')
print(f'Mean distance: {DF_DRN_all_matched["distance_to_match"].mean():.0f}m')

# %%
# # Quick summary of matching results
# print(f"Results: {len(DF_DRN_all_matched):,} DRN points matched")
# print(f"Distance stats: mean={DF_DRN_all_matched['distance_to_match'].mean():.0f}m, "
#       f"perfect_matches={(DF_DRN_all_matched['distance_to_match'] == 0).sum():,}")
# print(DF_DRN_all_matched[['k', 'i', 'j', 'X', 'Y', 'Pkg1', 'Rcv_ID', 'distance_to_match']].head())

# %%
DF_DRN_all_matched['Pkd2'] = 'sfr'

# %%
DF_DRN_write = DF_DRN_all_matched[['Pkg1', 'Pvd_ID', 'Pkd2', 'Rcv_ID']]
DF_DRN_write['MVR_TYPE'] = 'FACTOR'
DF_DRN_write['value'] = 1
DF_DRN_write

# %% [markdown]
# ### 4.3.2 Write MVR file

# %%
Pa_MVR = PJ(d_Pa['Sim_In'], f'{MdlN}.MVR6')

# %%
with open(Pa_MVR, 'w') as f:
    f.write(f"""BEGIN OPTIONS
END OPTIONS

BEGIN DIMENSIONS
    MAXMVR {DF_DRN_write.shape[0]}
    MAXPACKAGES {len(d_DRN_DF.keys()) + 1}
END DIMENSIONS

BEGIN PACKAGES
    {'\n    '.join([f'drn-{k}' for k in d_DRN_DF.keys()])}
    sfr
END PACKAGES

BEGIN PERIOD 1
""")
    f.write(U.DF_to_MF_block(DF_DRN_write))
    f.write('END PERIOD')

# %%
# Insert MVR line to NAM
with open(d_Pa['NAM_Mdl'], 'r') as f1:
    l_Lns_NAM = f1.readlines()

l_Lns_NAM.insert(-1, f'  MVR6 imported_model/{PBN(Pa_MVR)} MVR\n')

with open(d_Pa['NAM_Mdl'], 'w') as f2:
    f2.writelines(l_Lns_NAM)

# %%
# Add MOVER option to SFR
with open(d_Pa['SFR'], 'r') as f1:
    l_Lns_SFR = f1.readlines()

l_Lns_SFR.insert(3, '  MOVER\n')

with open(d_Pa['SFR'], 'w') as f2:
    f2.writelines(l_Lns_SFR)

# %%
# Add MOVER option to DRN files
for i in d_DRN_DF.keys():
    with open(PJ(d_Pa['Sim_In'], f'drn-{i}.drn'), 'r') as f1:
        l_Lns_DRN = f1.readlines()

    l_Lns_DRN.insert(3, '  MOVER\n')

    with open(PJ(d_Pa['Sim_In'], f'drn-{i}.drn'), 'w') as f2:
        f2.writelines(l_Lns_DRN)
print('🟢 - DRN connected to SFR via MVR.')
# endregion

# %% [markdown]
# region Add OBS
# # 5. Add HD OBS

# %%
# sh.copy2(Pa_HD_OBS, d_Pa['Sim_In']) # Adding HD OBS didn't work cause there are some cells outside the active domain. This throws up an error.

# %%
# U.add_PKG_to_NAM(MdlN=MdlN, str_PKG=f' OBS6 ./imported_model/NBr34.HD.OBS6 OBS_HD\n')

# %% [markdown]
# # 6. Add RIV OBS

# %%
sh.copy2(Pa_RIV_OBS, PJ(d_Pa['Sim_In'], PBN(Pa_RIV_OBS)))

# %%
U.add_OBS_to_MF_In(Pa=PJ(d_Pa['Sim_In'], PBN('rivriv.riv')), str_OBS=' OBS6 FILEIN ./imported_model/NBr45.RIV.OBS6')
print('🟢 - RIV OBS added to model.')
# endregion

# %% [markdown]
# region Correct mete_grid paths
# # 7. Correct mete_grip paths

# %%
U.mete_grid_add_missing_Cols(PJ(d_Pa['Pa_MdlN'], 'metaswap/mete_grid.inp'))
print('🟢 - mete_grid.inp corrected.')
# endregion
