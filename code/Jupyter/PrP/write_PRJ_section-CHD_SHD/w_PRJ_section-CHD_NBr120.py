# %% Imports
from datetime import datetime as DT
 
import pandas as pd
 
# %% Options
MdlN = 'NBr120'                 # simulation this block is written for (only used in the output file name)
MdlN_CHD = 'NBr120'             # simulation number the climatology CHD files come from (folder + file names)
date_start = '2036-01-01'
date_end = '2065-12-31'
n_layers = 37                   # total number of model layers; only the odd layers are written
 
# %% Build the list of CHD dates (14th and 28th of every month in the run period)
all_days = pd.date_range(date_start, date_end, freq='D')
dates = all_days[all_days.day.isin([14, 28])]
N_entries = len(dates)
layers = list(range(1, n_layers + 1, 2))   # 1, 3, 5, ... 37
 
 
# %% Write block/txt file
if True:
    with open(f'CHD_block_{DT.now().strftime("%Y_%m_%d")}_{MdlN}.txt', 'w') as f:
        f.write(f'{N_entries},(CHD),1, Constant Head Boundary')
        f.write('\n')
 
        for date in dates:
            f.write(date.strftime('%Y-%m-%d %H:%M:%S'))
            f.write('\n')
            f.write(f'{1:03d},{len(layers):03d}')
            f.write('\n')
            for j in layers:
                # climatology file: DDMM only, so every year points to the same file
                f.write(
                    rf" 1,2, {str(j).zfill(3)},   1.000000    ,   0.000000    ,  -999.9900    , '..\..\In\CHD\{MdlN_CHD}\CL_HD_{date.strftime('%d%m')}_L{j}_{MdlN_CHD}.idf' >>> (chd) constant head (idf) <<<"
                )
                f.write('\n')
 
# %%
