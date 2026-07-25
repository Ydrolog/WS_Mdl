"""---- Plot Layer T and Thk ----"""  # To decide which Ls are important/to use for OBS file creation

# %% Imports
import imod
import pandas as pd
import xarray as xra
from WS_Mdl.core import *
from WS_Mdl.imod.sfr.info import SFR_PkgD_to_DF

# %% Options
MdlN = 'NBr107'
M = Mdl_N(MdlN)

# %% Load T and Thk arrays
T = xra.open_dataset(r'G:\models\NBr\PoP\Clc_In\T\NBr1\T_NBr1.tif')
Thk = xra.open_dataset(r'G:\models\NBr\PoP\Clc_In\Thk\NBr1\Thk_NBr1.tif')
T['band_data'].attrs.clear()
Thk['band_data'].attrs.clear()

# %% Quickly check which Ls are "important"
print(f'{"L":3s} | {"T_min":7s} | {"T_max":7s} | {"Thk_min":7s} | {"Thk_max":7s}')

for L in T.band:
    T_min = float(T.sel(band=L).band_data.min().values)
    T_max = float(T.sel(band=L).band_data.max().values)
    Thk_min = float(Thk.sel(band=L).band_data.min().values)
    Thk_max = float(Thk.sel(band=L).band_data.max().values)

    if ((T_min < 0.05) and (T_max < 0.05)) or ((Thk_min < 0.01) and (Thk_max < 0.01)):
        print(f'{L:3} | {"-" * 7:7s} | {"-" * 7:7s} | {"-" * 7:7s} | {"-" * 7:7s}')
    else:
        print(f'{L:3} | {T_min:>7.2f} | {T_max:>7.2f} | {Thk_min:>7.2f} | {Thk_max:>7.2f}')

# %% Load SFR Ls
DF_SFR = SFR_PkgD_to_DF('NBr100')
sorted(DF_SFR.k.unique())
DF_SFR.k.value_counts()

# %%
"""
Now that I've decided which Ls to OBS, let's make the OBS file.
"""

# %% Options for writing the OBS file
Opt = """BEGIN OPTIONS\n  DIGITS 5\n  PRINT_INPUT\nEND OPTIONS\n"""
l_L = [
    1,
    3,
    5,
    7,
    9,
    10,
    11,
    13,
    23,  # High T in catchment
    35,  # High T thick and deep
    37,  # High T thick and deep
]

# %% Load GRB
ID = imod.mf6.read_grb(M.Pa_B.GRB)['idomain']

# %% Convert XY to R and C
ID = ID.rename({'y': 'R', 'x': 'C', 'layer': 'L'}).assign_coords(
    R=('R', ((ID.y[0] - ID.y) / M.cellsize + 1).astype(int).values),
    C=('C', ((ID.x - ID.x[0]) / M.cellsize + 1).astype(int).values),
)

ID = ID.sel(L=l_L)  # select only the Ls that I want to OBS

# %%
LRC = (
    ID.where(ID == 1, drop=True)
    .stack(cell=('L', 'R', 'C'))
    .dropna('cell')
    .cell.to_index()
    .map(lambda x: ' '.join(map(str, x)))
)

# %% Create DF for OBS file
DF = pd.DataFrame({'obsname': LRC.map(lambda x: 'HD_' + x.replace(' ', '_')), 'obstype': 'HEAD', 'id': LRC})

# %% Read Prvious OBS file to copy OBS Pnt block
MdlN_Prv = 'NBr99'
Pa_Prv = (M.Pa.In / f'OBS/HD/{MdlN_Prv}').glob('*.OBS6').__next__()

with open(Pa_Prv, 'r') as f:
    Prv = f.read()

block_to_copy = Prv.split('END CONTINUOUS', 1)[0].split('BEGIN CONTINUOUS', 1)[1]

# %% Write OBS file
Pa_OBS = M.Pa.In / f'OBS/HD/{MdlN}/HD_{M.MdlN}.OBS6'
Pa_OBS.parent.mkdir(parents=True, exist_ok=True)
with open(Pa_OBS, 'w') as f:
    f.write(f'# created by {__file__} using: {M.Pa_B.GRB}\n')
    f.write(Opt)  # write optional block
    f.write('\nBEGIN CONTINUOUS')
    f.write(block_to_copy)  # write OBS Pnt block
    f.write('END CONTINUOUS\n')
    f.write('\nBEGIN CONTINUOUS FILEOUT ./Out/HD_OBS_L.bin BINARY\n')
    f.write(DF.ws.to_MF_block())
    f.write('END CONTINUOUS\n')
