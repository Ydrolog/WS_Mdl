# %% Imports
import pandas as pd
from WS_Mdl.core import *

# %% Prep
MdlN = 'NBr105'
MdlN_B = 'NBr104'
M = Mdl_N(MdlN)
MB = Mdl_N(MdlN_B)  # B just for stages.
start_year = M.SP_1st_DT.year
end_year = 2001

# %% Load + Prep DF
Pa_SFR_Out = MB.Pa.Sim_In / f'{MB.MdlN}.SFR6.obs.output.csv'
DF = pd.read_csv(Pa_SFR_Out, engine='pyarrow')
DF['date'] = M.SP_1st_DT + pd.to_timedelta(DF['time'] - 1, unit='D')
DF = DF.loc[DF.date.dt.year <= end_year]

# %% Calc AVGs
Cols = [c for c in DF.columns if c.startswith('STAGE_L')]
DF_AVG = DF[Cols].mean()
DF_AVG = pd.DataFrame({'ID': DF_AVG.index.str.replace('STAGE_', ''), 'Stg_AVG': DF_AVG.values}).set_index('ID')

# %% Calc XY
DF_AVG[['L', 'R', 'C']] = DF_AVG.index.to_series().str.extract(r'L(\d+)_R(\d+)_C(\d+)').astype(int)
DF_AVG = DF_AVG.ws.Calc_XY(MB.Xmin, MB.Ymax, MB.cellsize)

# %% Convert to DA
DA = DF_AVG.set_index(['y', 'x'])[['Stg_AVG']].to_xarray()


# %% Load RIV
import imod
import xarray as xra
from WS_Mdl.xr.spatial import clip_Mdl_area

RIV_Stg = clip_Mdl_area(imod.idf.open(M.Pa.Mdl / r'In\RIV\RIV_Stg_Detailwatergangen_NBr1.IDF'), M.MdlN)

# %% Align Coordinates, Join, Save
Stg_aligned, riv_aligned = xra.align(
    DA['Stg_AVG'],
    RIV_Stg,
    join='outer',
)

# %% Combine
RIV_Stg_Out = xra.where(Stg_aligned.notnull(), Stg_aligned, riv_aligned)

# %% Save
imod.idf.save(
    M.Pa.In / f'RIV/{MdlN}/RIV_Stg_detail_{M.MdlN}.IDF',
    RIV_Stg_Out,
)
