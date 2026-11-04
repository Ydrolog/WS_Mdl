# %% [markdown]
# # 1. Imports
import os

import geopandas as gpd
import pandas as pd
import WS_Mdl.core.df  # Noqa: F401
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.imod.mf6.headers import d_Pkg_Cols

# %% [markdown]
# # 2. Options
MdlN = 'NBr128'
M = Mdl_N(MdlN)

# %% [markdown]
# # 3. Read in list of DRN cells
l_Pa_DRN = list(M.Pa.Sim_In.rglob('*drn*.dat'))

# %%
DF_DRN = pd.DataFrame()
for i in l_Pa_DRN:
    DF = pd.read_csv(i, sep=r'\s+', skiprows=1, names=d_Pkg_Cols['DRN'])
    DF['system'] = i.parent.name.split('-')[-1][0]
    DF_DRN = pd.concat([DF_DRN, DF], ignore_index=True)
DF_DRN_ = DF_DRN.copy()  # Copy to avoid modifying OG
DF_DRN

# %%
# DF_DRN.loc[(DF_DRN['stage'] - DF_DRN['rbot']) > 0.01].empty
DF_DRN.rename(columns={'k': 'L', 'i': 'R', 'j': 'C'}, inplace=True)
DF_DRN = DF_DRN.drop_duplicates(subset=['L', 'R', 'C'])  # Keep first occurrence

# %%
DF_DRN = DF_DRN.drop(
    columns=[i for i in DF_DRN.columns if i not in ['L', 'R', 'C']]
)  # We only need L, R, C for matching.

DF_DRN.index += 1  # Make it 1-based index, as in MODFLOW.

# %%
DF_DRN = DF_DRN.reset_index().rename(columns={'index': 'Pvd_i'})
DF_DRN.tail()

# %% [markdown]
# # 4. L R C to X Y
DF_DRN = DF_DRN.ws.Calc_XY(Xmin=M.Xmin, Ymax=M.Ymax, cellsize=M.cellsize)  # Calc X & Y

# %% [markdown]
# ## 5. Load Chaamse Beek shapefile and limit DFs to it's extent
Pa_CB = M.Pa.PoP / r'common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp'  # CB: Chaamse Beek
GDF_CB = gpd.read_file(Pa_CB)
GDF_CB.crs = CRS  # Set CRS to match model's CRS
print(f'Loaded shapefile with {len(GDF_CB)} features')
print(f'Bounds: {GDF_CB.bounds}')

# %%
from shapely.geometry import Point

# %%
# Create geometry for DRN points
DF_DRN['geometry'] = DF_DRN.apply(lambda row: Point(row['x'], row['y']), axis=1)
GDF_DRN = gpd.GeoDataFrame(DF_DRN, crs=CRS)

print(f'Created GeoDataFrame for DRN with {len(GDF_DRN)} points')
print(
    f'DRN points bounds: minX={GDF_DRN.bounds["minx"].min():.2f}, minY={GDF_DRN.bounds["miny"].min():.2f}, maxX={GDF_DRN.bounds["maxx"].max():.2f}, maxY={GDF_DRN.bounds["maxy"].max():.2f}'
)

# %%
# Store original counts before filtering
original_drn_count = len(GDF_DRN)

# Perform spatial intersection to limit points to catchment extent
GDF_DRN_in = gpd.sjoin(GDF_DRN, GDF_CB, how='inner', predicate='within')

print(
    f'DRN points within catchment: {len(GDF_DRN_in)} out of {original_drn_count} ({len(GDF_DRN_in) / original_drn_count * 100:.1f}%)'
)

# %%
# Update the original dataframes with only the points within the catchment
# Remove the extra columns from the spatial join (keep only original columns)
original_drn_cols = ['Pvd_i', 'L', 'R', 'C', 'x', 'y']
original_drn_cols = ['Pvd_i', 'L', 'R', 'C', 'x', 'y']

DF_DRN_in = GDF_DRN_in[original_drn_cols].copy()

print('Final limited dataframes:')
print(
    f'DF_DRN_in: {len(DF_DRN_in)} points, out of original {len(DF_DRN_)} points -> {len(DF_DRN_in) / len(DF_DRN_) * 100:.1f}%'
)

# %% [markdown]
# # 6. Build OBS files.
Dir_DRN_OBS = M.Pa.In / f'OBS/DRN/{MdlN}'
os.makedirs(Dir_DRN_OBS, exist_ok=True)

# %%
Pa_DRN_OBS = Dir_DRN_OBS / f'{MdlN}.DRN.OBS'

# %%
Opt = """BEGIN OPTIONS
	DIGITS 4
	PRINT_INPUT
END OPTIONS\n
"""

# %%
DF_DRN_w = pd.DataFrame()
DF_DRN_w['obsname'] = DF_DRN_in.apply(lambda row: f'DRN_L{int(row["L"])}_R{int(row["R"])}_C{int(row["C"])}', axis=1)
DF_DRN_w['obstype'] = 'drn'
DF_DRN_w['id'] = DF_DRN_in.apply(lambda row: f'{int(row["L"])} {int(row["R"])} {int(row["C"])}', axis=1)

# %%
with open(Pa_DRN_OBS, 'w') as f:
    f.write(Opt)
    f.write('BEGIN CONTINUOUS FILEOUT DRN_OBS.CSV\n')
    f.write(DF_DRN_w.ws.to_MF_block())
    f.write('END CONTINUOUS FILEOUT\n')

# %%
DF_DRN_in.sort_values(by=['R', 'C', 'L'], inplace=True)

# %%
DF_DRN_in
