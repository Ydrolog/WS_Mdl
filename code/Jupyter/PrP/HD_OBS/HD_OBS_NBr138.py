"""---- Plot Layer T and Thk ----"""  # To decide which Ls are important/to use for OBS file creation

# %% Imports
import imod
import pandas as pd
import WS_Mdl.core.df  # noqa: F401
from WS_Mdl.core.mdl import Mdl_N

# %% Options
MdlN = 'NBr138'
MdlN_B = 'NBr137'

M = Mdl_N(MdlN)
MB = Mdl_N(MdlN_B, iMOD5=True)

# %% Options for writing the OBS file
Opt = """BEGIN OPTIONS\n  DIGITS 5\n  PRINT_INPUT\nEND OPTIONS\n"""
l_L = [  # First few layers have adeqate Thk and are connected to the SFR.
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
ID = imod.mf6.read_grb(next(iter(MB.Pa.MdlN.rglob('*.grb'))))['idomain']

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

# # %% Read Prvious OBS file to copy OBS Pnt block
# MdlN_Prv = 'NBr107'
# Pa_Prv = next((M.Pa.In / f'OBS/HD/{MdlN_Prv}').glob('*.OBS*'))

# with open(Pa_Prv, 'r') as f:
#     Prv = f.read()

# block_to_copy = Prv.split('END CONTINUOUS', 1)[0].split('BEGIN CONTINUOUS', 1)[1]

# %% Write OBS file
Pa_OBS = M.Pa.In / f'OBS/HD/{MdlN}/HD_{M.MdlN}.OBS'
Pa_OBS.parent.mkdir(parents=True, exist_ok=True)
with open(Pa_OBS, 'w') as f:
    f.write(f'# created by {__file__} using: {MB.Pa_B.GRB}\n')
    f.write(Opt)  # write optional block
    # f.write('\nBEGIN CONTINUOUS')
    # f.write(block_to_copy)  # write OBS Pnt block
    # f.write('END CONTINUOUS\n')
    f.write('\nBEGIN CONTINUOUS FILEOUT ./Out/HD_OBS_L.bin BINARY\n')
    f.write(DF.ws.to_MF_block())
    f.write('END CONTINUOUS\n')

# %%
