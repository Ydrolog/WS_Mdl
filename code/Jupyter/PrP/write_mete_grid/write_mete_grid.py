# %% [markdown]
# <span style="font-size:24px; font-family:'Roboto'; font-weight:bold;">
# Script to write a mete_grid.inp file
# </span><br>
# Will read inputs from write_mete_grid_In.txt to write the mete_grid.inp, which will be placed in the corresponding folder.

# %% [markdown]
# ## 1. Options

# %%
import pandas as pd
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
from datetime import datetime as DT

# %%
with open('./write_mete_grid_In.txt', "r") as file:
    params = {}
    exec(file.read(), {}, params)

# Access parameters
date_start = params["date_start"]
date_end = params["date_end"]
Mdl = params["Mdl"]
SimN = params["SimN"] #SimN of mete_grid file
SimN_P = params["SimN_P"] #SimN of P. P and PET grids can belong to a previous run. 
SimN_PET = params["SimN_PET"] #SimN of PET. P and PET grids can belong to a previous run. 

print('Writing mete_grid.py with input parameters:')
for p in params:
    print(p, params[p])

# %% [markdown]
# ## 2 Read and prep DF

# %%
DF = pd.read_csv(r'../../../data/Dates.csv')

# %%
DF['Date'] = pd.to_datetime(DF['Date'])

# %%
DF = DF.loc[(DF['Date'] >= date_start) & (DF['Date'] <= date_end)].reset_index(drop=True)

# %%
DF['Year'] = DF['Date'].dt.year
DF['Month'] = DF['Date'].dt.month
DF['Day'] = DF['Date'].dt.day
DF['DayOfYear'] = DF['Date'].dt.dayofyear-1

# %% [markdown]
# ## 3. Write block/txt file

# %%
with open(f'../../../models/{Mdl}/In/CAP/mete_grid/{Mdl+str(SimN)}/mete_grid.inp', 'w') as f:
    for i, row in DF.iterrows():
        f.write(rf'{row["DayOfYear"]:.2f},{row["Year"]},"..\..\In\CAP\P\{Mdl+str(SimN_P)}\P_{row["Date"].strftime("%Y%m%d")}_{Mdl+str(SimN_P)}.asc","..\..\In\CAP\PET\{Mdl+str(SimN_PET)}\PET_{row["Date"].strftime("%Y%m%d")}_{Mdl+str(SimN_PET)}.asc"')
        f.write('\n')


