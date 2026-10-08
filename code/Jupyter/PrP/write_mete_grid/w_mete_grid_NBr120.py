# %% Markdown cell
# Script to write a mete_grid.inp from TWO sources:
# days < date_switch use the OLD folders (e.g. NBr1), days >= date_switch use the NEW folders (e.g. NBr120).
# %% Imports
import os
import sys
import pandas as pd
 
# %% Options
date_start = '1991-01-01' # start date of the simulation
date_end = '2020-12-31'   # end date of the simulation
date_switch = '2020-10-06'  # first day taken from the NEW folders
Mdl = 'NBr' # short for Noord-Brabant-run 
SimN = 120  # SimN of the mete_grid file that is written
SimN_P_old = 1 
SimN_PET_old = 1
SimN_P_new = 120
SimN_PET_new = 120
check_files = True  # stop if a referenced .asc does not exist
 
ROOT = r'../../../..'  
 
# %% Read and prep DF
DF = pd.read_csv(f'{ROOT}/data/Dates.csv')
DF['Date'] = pd.to_datetime(DF['Date'])
DF = DF.loc[(DF['Date'] >= date_start) & (DF['Date'] <= date_end)].reset_index(drop=True)
DF['Year'] = DF['Date'].dt.year
DF['DayOfYear'] = DF['Date'].dt.dayofyear - 1
 
# date list must be complete, and the switch must lie inside it
if not (DF['Date'].diff().dropna() == pd.Timedelta(days=1)).all():
    sys.exit('Dates.csv has gaps or duplicates in the selected period.')
switch = pd.Timestamp(date_switch)
if not (DF['Date'].min() < switch <= DF['Date'].max()):
    sys.exit('date_switch lies outside the period: nothing would be mixed.')
 
# %% Pick the folder per day
new = DF['Date'] >= switch
DF['SimP'] = [Mdl + str(SimN_P_new if n else SimN_P_old) for n in new]
DF['SimPET'] = [Mdl + str(SimN_PET_new if n else SimN_PET_old) for n in new]
DF['ds'] = DF['Date'].dt.strftime('%Y%m%d')
 
# %% Check that every .asc exists 
if check_files:
    missing = []
    for kind, col in (('P', 'SimP'), ('PET', 'SimPET')):
        folder_files = {}
        for sim in DF[col].unique():
            d = f'{ROOT}/models/{Mdl}/In/CAP/{kind}/{sim}'
            folder_files[sim] = set(os.listdir(d)) if os.path.isdir(d) else set()
            if not os.path.isdir(d):
                print('Folder not found:', d)
        for sim, ds in zip(DF[col], DF['ds']):
            fn = f'{kind}_{ds}_{sim}.asc'
            if fn not in folder_files[sim]:
                missing.append(f'{kind}/{sim}/{fn}')
    if missing:
        print(f'{len(missing)} referenced files do NOT exist, first 10:')
        for m in missing[:10]:
            print('  ', m)
        sys.exit('mete_grid.inp NOT written.')
    print('All referenced files exist.')
 
# %% Write mete_grid.inp
out_dir = f'{ROOT}/models/{Mdl}/In/CAP/mete_grid/{Mdl}{SimN}'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/mete_grid.inp', 'w') as f:
    for _, row in DF.iterrows():
        f.write(
            rf'{row["DayOfYear"]:.2f},{row["Year"]},"..\..\In\CAP\P\{row["SimP"]}\P_{row["ds"]}_{row["SimP"]}.asc","..\..\In\CAP\PET\{row["SimPET"]}\PET_{row["ds"]}_{row["SimPET"]}.asc"'
        )
        f.write('\n')
 
print(f'Wrote {len(DF)} lines to {out_dir}/mete_grid.inp: {(~new).sum()} days old grids, {new.sum()} days new grids.')