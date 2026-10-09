# %% [markdown]
# Script to write a mete_grid.inp from TWO sources, per variable (P and PET separately):
# days inside one of the *_new_ranges use the NEW folder (e.g. NBr120), all other days the OLD folder (e.g. NBr1).
# Run it with the working directory = the folder of this script (as with the other scripts).

# %% Imports
import os
import sys

import pandas as pd

# %% Options
date_start = '1991-01-01'
date_end = '2020-12-31'
# Periods (inclusive, 'start','end') that are taken from the NEW folders. Everything else comes from the OLD folders.
P_new_ranges = [('2020-10-06', '2020-12-31')]
PET_new_ranges = [('2008-12-31', '2009-12-31'),   
                  ('2016-12-31', '2016-12-31'),   
                  ('2020-10-06', '2020-12-31')]   
Mdl = 'NBr'
SimN = 120  # SimN of the mete_grid file that is written
SimN_P_old = 1
SimN_PET_old = 1
SimN_P_new = 120
SimN_PET_new = 120
check_files = True  # stop if a referenced .asc does not exist

ROOT = r'../../../..'  # from this script's folder to the project root (the folder with data/ and models/)

# %% Read and prep DF
DF = pd.read_csv(f'{ROOT}/data/Dates.csv')
DF['Date'] = pd.to_datetime(DF['Date'])
DF = DF.loc[(DF['Date'] >= date_start) & (DF['Date'] <= date_end)].reset_index(drop=True)
DF['Year'] = DF['Date'].dt.year
DF['DayOfYear'] = DF['Date'].dt.dayofyear - 1

# date list must be complete
if not (DF['Date'].diff().dropna() == pd.Timedelta(days=1)).all():
    sys.exit('Dates.csv has gaps or duplicates in the selected period.')


def in_ranges(dates, ranges):
    m = pd.Series(False, index=dates.index)
    for a, b in ranges:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a < DF['Date'].min() or b > DF['Date'].max() or a > b:
            sys.exit(f'Range {a.date()}..{b.date()} is invalid or outside the period '
                     f'{DF["Date"].min().date()}..{DF["Date"].max().date()}.')
        m |= (dates >= a) & (dates <= b)
    return m


# %% Pick the folder per day and per variable
newP = in_ranges(DF['Date'], P_new_ranges)
newPET = in_ranges(DF['Date'], PET_new_ranges)
DF['SimP'] = [Mdl + str(SimN_P_new if n else SimN_P_old) for n in newP]
DF['SimPET'] = [Mdl + str(SimN_PET_new if n else SimN_PET_old) for n in newPET]
DF['ds'] = DF['Date'].dt.strftime('%Y%m%d')

# %% Check that every .asc exists (one folder listing per folder; fast on network drives)
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

print(f'Wrote {len(DF)} lines to {out_dir}/mete_grid.inp')
print(f'  P:   {(~newP).sum()} days old folder, {newP.sum()} days new folder')
print(f'  PET: {(~newPET).sum()} days old folder, {newPET.sum()} days new folder')
