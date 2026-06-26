# %% Options
MdlN = 'NBr100'
SP = 0  # 0 based indexing

# %% Init
import pandas as pd
from WS_Mdl.core.mdl import Mdl_N

M = Mdl_N(MdlN)
MB = Mdl_N(M.B)

# %% Load Out
DF_full = pd.read_csv(r'g:\models\NBr\Sim\NBr99\modflow6\imported_model\NBr99.SFR6.obs.output.csv')

# %% Slice
Cols = [i for i in DF_full.columns if 'stage_l' in i.lower()]
DF = DF_full.loc[:, Cols].iloc[SP : SP + 1].T
DF.rename(columns={DF.columns[0]: 'Stg'}, inplace=True)

# %% Prepare ID Col
DF['ID'] = DF.index.map(lambda x: x.replace('STAGE_', ''))

# %% Load Package Data & convert ID to rno
from WS_Mdl.imod.sfr.info import SFR_PkgD_to_DF

DF_ID = SFR_PkgD_to_DF(M.B).loc[:, ['rno', 'k', 'i', 'j']]
DF_ID['ID'] = DF_ID.apply(lambda row: f'L{row["k"]}_R{row["i"]}_C{row["j"]}', axis=1)
DF_ID.drop(columns=['k', 'i', 'j'], inplace=True)

# %% Merge Data
DF_Out = DF_ID.merge(DF, on='ID', how='outer').drop(columns=['ID'])

# %%
date = MB.SP_1st_DT + pd.Timedelta(days=SP)

# %% Save
Pa = M.Pa.In / f'SFR/Stg_Init/{MdlN}/Stg_Init_{date.strftime("%Y%m%d")}_{MdlN}.csv'
Pa.parent.mkdir(parents=True, exist_ok=True)
DF_Out.to_csv(Pa, index=False)

# %% write metadata
with open(Pa.parent / f'SFR_Stg_Init_{MdlN}.txt', 'w') as f:
    f.write(
        f'Created on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\nBy using stages from {M.B} model\nFor {MdlN} model\nSP: {SP}\nDate: {date.strftime("%Y-%m-%d")}\n'
    )
