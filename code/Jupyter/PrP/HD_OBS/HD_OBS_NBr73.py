# %% Init
import imod
import pandas as pd
from WS_Mdl.core import *

# %%
Opt = """BEGIN OPTIONS\n  DIGITS 4\n  PRINT_INPUT\nEND OPTIONS\n\n"""
l_L = [1, 3, 5, 7, 9, 11]
M = Mdl_N('NBr73')

# %%
ID = imod.mf6.read_grb(M.Pa_B.GRB)['idomain']

# %%
ID_ = ID.rename({'y': 'R', 'x': 'C', 'layer': 'L'}).assign_coords(
    R=('R', ((ID.y[0] - ID.y) / M.cellsize + 1).astype(int).values),
    C=('C', ((ID.x - ID.x[0]) / M.cellsize + 1).astype(int).values),
)

# %%
LRC = (
    ID_.where(ID_ == 1, drop=True)
    .stack(cell=('L', 'R', 'C'))
    .dropna('cell')
    .cell.to_index()
    .map(lambda x: ' '.join(map(str, x)))
)

# %% Write OBS file
DF = pd.DataFrame({'obsname': LRC.map(lambda x: 'HD_' + x.replace(' ', '_')), 'obstype': 'HEAD', 'id': LRC})
with open(M.Pa.Sim_In / f'L_HD_OBS_{M.MdlN}.OBS6', 'w') as f:
    f.write(f'# created with {M.Pa_B.GRB}\n')
    f.write(Opt)  # write optional block
    f.write(f'\n\nBEGIN CONTINUOUS FILEOUT L_HD_OBS_{M.MdlN}.csv\n')
    f.write(DF.ws.to_MF_block())
    f.write('END CONTINUOUS\n')
