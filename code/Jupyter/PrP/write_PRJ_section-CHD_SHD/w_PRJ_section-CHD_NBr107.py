# %% Imports
from datetime import datetime as DT

import pandas as pd

# %% Options
MdlN = 'NBr107'
date_start = '2010-01-01'
date_end = '2018-12-31'

# %% Read and prep DF
DF = pd.read_csv(r'../../../../data/Dates.csv')
DF['Date'] = pd.to_datetime(DF['Date'])
DF = DF.loc[(DF['Date'] >= date_start) & (DF['Date'] <= date_end)].reset_index(drop=True)
DF['Year'] = DF['Date'].dt.year
DF['Month'] = DF['Date'].dt.month
DF['Day'] = DF['Date'].dt.day
DF_14 = DF.loc[DF['Day'] == 14]
N_entries = DF_14.shape[0] + 1


# %% Write block/txt file
if True:
    with open(f'CHD_block_{DT.now().strftime("%Y_%m_%d")}_{MdlN}.txt', 'w') as f:
        f.write(f'{N_entries},(CHD),1, Constant Head Boundary')
        f.write('\n')

        # Write first day of Sim
        date_start = '2010-01-14'  # There is no CHD for he first date in this dataset, so I'm using the 2nd one for the 1st SP too.
        f.write(f'{date_start} 00:00:00')
        date = DT.strptime(date_start, '%Y-%m-%d')
        f.write('\n')
        f.write('001,019')
        f.write('\n')
        for j in range(1, 37 + 1, 2):
            f.write(
                rf" 1,2, {str(j).zfill(3)},   1.000000    ,   0.000000    ,  -999.9900    , '..\..\In\CHD\NBr5\head_{date.strftime('%Y%m%d')}_L{j}.idf' >>> (chd) constant head (idf) <<<"
            )
            f.write('\n')

        # The rest of the blocks every 14th of each month (as in the OG model)
        for i, date in DF_14['Date'].items():
            f.write(date.strftime('%Y-%m-%d %H:%M:%S'))
            f.write('\n')
            f.write('001,019')
            f.write('\n')
            for j in range(1, 37 + 1, 2):
                f.write(
                    rf" 1,2, {str(j).zfill(3)},   1.000000    ,   0.000000    ,  -999.9900    , '..\..\In\CHD\NBr5\head_{date.strftime('%Y%m%d')}_L{j}.idf' >>> (chd) constant head (idf) <<<"
                )
                f.write('\n')

# %%
