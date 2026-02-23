# %% [markdown]
# # 0. Basics

# %% [markdown]
# ## 0.0. Imports

# %%
# %%
import importlib as IL
import os
from datetime import datetime as DT
from os.path import dirname as PDN
from os.path import join as PJ

import pandas as pd

# %%
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM

IL.reload(U)
IL.reload(UIM)

# %%
# Import sfrmaker and other necessary packages for SFR network creation
import geopandas as gpd
import numpy as np

# %%
# %%
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)

# %% [markdown]
# ## 0.1. Options

# %%
MdlN = 'NBr51'

# %%
U.set_verbose(False)

# %%
# Load paths and variables from PRJ & INI
d_Pa = U.get_MdlN_Pa(MdlN)
Pa_PRJ = d_Pa['PRJ']
Dir_PRJ = PDN(Pa_PRJ)
d_INI = U.INI_to_d(d_Pa['INI'])
Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
SP_date_1st, SP_date_last = [DT.strftime(DT.strptime(d_INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']]
dx = dy = float(d_INI['CELLSIZE'])

# %% [markdown]
# # 1. Read in list of RIV cells

# %% [markdown]
# ## 1.0. Read in

# %%
Pa_RIV = PJ(d_Pa['Sim_In'], 'rivriv/riv-0.bin')
Pa_RIV

# %%
from pathlib import Path


def read_mf6_riv_bin(filepath: str | Path) -> pd.DataFrame:
    """Read MODFLOW 6 RIV binary input (imod format) into a DataFrame."""
    dtype = np.dtype(
        [
            ('k', '<i4'),  # layer
            ('i', '<i4'),  # row
            ('j', '<i4'),  # column
            ('stage', '<f8'),  # stage
            ('cond', '<f8'),  # conductance
            ('rbot', '<f8'),  # river bottom
        ]
    )
    path = Path(filepath)
    nrec = path.stat().st_size // dtype.itemsize
    arr = np.fromfile(path, dtype=dtype, count=nrec)
    return pd.DataFrame(arr)


DF_RIV = read_mf6_riv_bin(Pa_RIV)  # .replace('NBr45', 'NBr44'))
DF_RIV.head()

# %%
DF_RIV

# %% [markdown]
# ## 1.1. Explore & edit.

# %%
DF_RIV

# %%
DF_RIV.loc[(DF_RIV['stage'] - DF_RIV['rbot']) > 0.01].empty

# %% [markdown]
# Elv is just Elv2 rounded to the 2nd decimal.

# %%
DF_RIV.rename(columns={'k': 'L', 'i': 'R', 'j': 'C'}, inplace=True)

# %%
DF_RIV_ = DF_RIV.copy()  # Copy to avoid modifying original DF_DRN.
DF_RIV_ = DF_RIV_.drop(
    columns=[i for i in DF_RIV_.columns if i not in ['L', 'R', 'C']]
)  # We only need L, R, C for matching.

# %%
DF_RIV_.index += 1  # Make it 1-based index, as in MODFLOW.

# %%
DF_RIV_ = DF_RIV_.reset_index().rename(columns={'index': 'Pvd_i'})

# %%
DF_RIV_.tail()

# %% [markdown]
# ## 1.2. L R C to X Y

# %%
# Add X and Y coordinates to RIV dataframe and create geometry
DF_RIV_['X'] = Xmin + DF_RIV_['C'] * dx - dx / 2
DF_RIV_['Y'] = Ymax - DF_RIV_['R'] * dy + dy / 2


# %% [markdown]
# ## 1.3. Load Chaamse Beek shapefile and limit DFs to it's extent

# %%
Pa_CB = PJ(U.Pa_WS, r'models\NBr\PoP\common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp')  # CB: Chaamse Beek
GDF_CB = gpd.read_file(Pa_CB)
print(f'Loaded shapefile with {len(GDF_CB)} features')
print(f'CRS: {GDF_CB.crs}')
print(f'Bounds: {GDF_CB.bounds}')

# %%
from shapely.geometry import Point

# %%
# Create geometry for RIV points
DF_RIV_['geometry'] = DF_RIV_.apply(lambda row: Point(row['X'], row['Y']), axis=1)
GDF_RIV = gpd.GeoDataFrame(DF_RIV_, crs=GDF_CB.crs)

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
original_drn_cols = ['Pvd_i', 'L', 'R', 'C', 'X', 'Y']
original_riv_cols = ['Pvd_i', 'L', 'R', 'C', 'X', 'Y']

DF_RIV_in = GDF_RIV_in[original_riv_cols].copy()

print('Final limited dataframes:')
print(
    f'DF_RIV_in: {len(DF_RIV_in)} points, out of original {len(DF_RIV_)} points -> {len(DF_RIV_in) / len(DF_RIV_) * 100:.1f}%'
)

# %% [markdown]
# # 2. Build OBS files.

# %% [markdown]
# ## 2.0. General

# %%
Dir_RIV_OBS = PJ(d_Pa['Pa_Mdl'], f'In/OBS/RIV/{MdlN}')
os.makedirs(Dir_RIV_OBS, exist_ok=True)

# %%
Pa_RIV_OBS = PJ(Dir_RIV_OBS, f'{MdlN}.RIV.OBS6')


# %%
Opt = """BEGIN OPTIONS
	DIGITS 4
	PRINT_INPUT
END OPTIONS\n
"""

# %% [markdown]
# ## 2.1. RIV OBS

# %%
DF_RIV_w = pd.DataFrame()
DF_RIV_w['obsname'] = DF_RIV_in.apply(lambda row: f'RIV_L{int(row["L"])}_R{int(row["R"])}_C{int(row["C"])}', axis=1)
DF_RIV_w['obstype'] = 'riv'
DF_RIV_w['id'] = DF_RIV_in.apply(lambda row: f'{int(row["L"])} {int(row["R"])} {int(row["C"])}', axis=1)

# %%
Pa_RIV_OBS

# %%
with open(Pa_RIV_OBS, 'w') as f:
    f.write(Opt)
    f.write('BEGIN CONTINUOUS FILEOUT RIV_OBS.CSV\n')
    f.write(U.DF_to_MF_block(DF_RIV_w))
    f.write('END CONTINUOUS FILEOUT\n')

# %%
PJ(d_Pa['MF6'], 'RIV_OBS.CSV')

# %%


# %%
DF_RIV_in.sort_values(by=['R', 'C', 'L'], inplace=True)

# %%
DF_RIV_in

# %%


# %%
# with open(PJ(d_Pa['Sim_In'], 'NBr44.RIV.OBS6')) as f:
#     content = f.readlines()

# %%
# content

# %%
# DF = pd.read_csv(PJ(d_Pa['Sim_In'], 'NBr44.RIV.OBS6'), skiprows=7, skipfooter=1, engine='python', delim_whitespace=True, names=['obsname', 'obstype', 'L', 'R', 'C'])

# %%
# DF.R.min(), DF.R.max()

# %%
# DF.sort_values(by=['R', 'C', 'L'])

# %% [markdown]
# # 3. Array analysis

# %%
# A = imod.idf.open(r'g:\models\NBr\In\RIV\NBr44\RIV_Stg_Detailwatergangen_NBr44.IDF')

# %%
# # Option 1: Convert to DataFrame to see coordinates (x, y, layer) and values
# df_A = A.to_dataframe().reset_index().dropna()
# display(df_A)

# # Option 2: Get integer indices (Row, Column) using numpy
# # This returns a tuple of arrays, one for each dimension
# import numpy as np
# indices = np.where(~np.isnan(A.values))
# # indices[0] are row indices, indices[1] are column indices (if 2D)

# %%
# df_A['R'] = (Ymax - df_A['y']) // dy + 1

# %%
# df_A['C'] = (df_A['x'] - Xmin) // dx + 1

# %%
# df_A
