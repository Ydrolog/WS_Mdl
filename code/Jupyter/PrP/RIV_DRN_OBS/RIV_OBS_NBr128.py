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
# # 3. Read in list of RIV cells
l_Pa_RIV = list(M.Pa.Sim_In.rglob('*riv*.dat'))

# %%
DF_RIV = pd.DataFrame()
for i in l_Pa_RIV:
    DF = pd.read_csv(i, sep=r'\s+', skiprows=1, names=d_Pkg_Cols['RIV'])
    DF['system'] = i.parent.name.split('-')[-1][0]
    DF_RIV = pd.concat([DF_RIV, DF], ignore_index=True)
DF_RIV_ = DF_RIV.copy()  # Copy to avoid modifying OG
DF_RIV

# %%
# DF_RIV.loc[(DF_RIV['stage'] - DF_RIV['rbot']) > 0.01].empty
DF_RIV.rename(columns={'k': 'L', 'i': 'R', 'j': 'C'}, inplace=True)
DF_RIV = DF_RIV.drop_duplicates(subset=['L', 'R', 'C'])  # Keep first occurrence

# %%
DF_RIV = DF_RIV.drop(
    columns=[i for i in DF_RIV.columns if i not in ['L', 'R', 'C']]
)  # We only need L, R, C for matching.

DF_RIV.index += 1  # Make it 1-based index, as in MODFLOW.

# %%
DF_RIV = DF_RIV.reset_index().rename(columns={'index': 'Pvd_i'})
DF_RIV.tail()

# %% [markdown]
# # 4. L R C to X Y
DF_RIV = DF_RIV.ws.Calc_XY(Xmin=M.Xmin, Ymax=M.Ymax, cellsize=M.cellsize)  # Calc X & Y

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
# Create geometry for RIV points
DF_RIV['geometry'] = DF_RIV.apply(lambda row: Point(row['x'], row['y']), axis=1)
GDF_RIV = gpd.GeoDataFrame(DF_RIV, crs=CRS)

print(f'Created GeoDataFrame for RIV with {len(GDF_RIV)} points')
print(
    f'RIV points bounds: minX={GDF_RIV.bounds["minx"].min():.2f}, minY={GDF_RIV.bounds["miny"].min():.2f}, maxX={GDF_RIV.bounds["maxx"].max():.2f}, maxY={GDF_RIV.bounds["maxy"].max():.2f}'
)

# %%
# Store original counts before filtering
original_riv_count = len(GDF_RIV)

# Perform spatial intersection to limit points to catchment extent
GDF_RIV_in = gpd.sjoin(GDF_RIV, GDF_CB, how='inner', predicate='within')

print(
    f'RIV points within catchment: {len(GDF_RIV_in)} out of {original_riv_count} ({len(GDF_RIV_in) / original_riv_count * 100:.1f}%)'
)

# %%
# Update the original dataframes with only the points within the catchment
# Remove the extra columns from the spatial join (keep only original columns)
original_drn_cols = ['Pvd_i', 'L', 'R', 'C', 'x', 'y']
original_riv_cols = ['Pvd_i', 'L', 'R', 'C', 'x', 'y']

DF_RIV_in = GDF_RIV_in[original_riv_cols].copy()

print('Final limited dataframes:')
print(
    f'DF_RIV_in: {len(DF_RIV_in)} points, out of original {len(DF_RIV_)} points -> {len(DF_RIV_in) / len(DF_RIV_) * 100:.1f}%'
)

# %% [markdown]
# # 6. Build OBS files.
Dir_RIV_OBS = M.Pa.In / f'OBS/RIV/{MdlN}'
os.makedirs(Dir_RIV_OBS, exist_ok=True)

# %%
Pa_RIV_OBS = Dir_RIV_OBS / f'{MdlN}.RIV.OBS'

# %%
Opt = """BEGIN OPTIONS
	DIGITS 4
	PRINT_INPUT
END OPTIONS\n
"""

# %%
DF_RIV_w = pd.DataFrame()
DF_RIV_w['obsname'] = DF_RIV_in.apply(lambda row: f'RIV_L{int(row["L"])}_R{int(row["R"])}_C{int(row["C"])}', axis=1)
DF_RIV_w['obstype'] = 'riv'
DF_RIV_w['id'] = DF_RIV_in.apply(lambda row: f'{int(row["L"])} {int(row["R"])} {int(row["C"])}', axis=1)

# %%
with open(Pa_RIV_OBS, 'w') as f:
    f.write(Opt)
    f.write('BEGIN CONTINUOUS FILEOUT RIV_OBS.CSV\n')
    f.write(DF_RIV_w.ws.to_MF_block())
    f.write('END CONTINUOUS FILEOUT\n')

# %%
DF_RIV_in.sort_values(by=['R', 'C', 'L'], inplace=True)

# %%
DF_RIV_in
