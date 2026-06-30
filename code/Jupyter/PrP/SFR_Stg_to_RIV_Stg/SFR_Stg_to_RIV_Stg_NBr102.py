# %% Imports
import pandas as pd
from WS_Mdl.core import *

# %% Prep
MdlN = 'NBr102'
MdlN_B = 'NBr100'
M = Mdl_N(MdlN)
MB = Mdl_N(MdlN_B)  # B just for stages.
start_year = M.SP_1st_DT.year
end_year = 2001

# %% Load + Prep DF
Pa_SFR_Out = MB.Pa.Sim_In / f'{MB.MdlN}.SFR6.obs.output.csv'
DF_ = pd.read_csv(Pa_SFR_Out)

# %%
date = MB.SP_1st_DT + pd.to_timedelta(DF_['time'] - 1, unit='D')

DF = pd.concat(
    [
        pd.DataFrame({'date': date, 'month': date.dt.month}, index=DF_.index),
        DF_.filter(like='STAGE_L'),
    ],
    axis=1,
)
DF = DF.loc[DF.date.dt.year <= end_year]

# %% Calc AVGs
Cols = [c for c in DF.columns if c.startswith('STAGE_L')]
DF_AVG = pd.DataFrame(
    {
        'AVG_Stg_summer': DF.loc[(DF['month'] > 3) & (DF['month'] < 10)].copy()[Cols].mean(axis='index'),
        'AVG_Stg_winter': DF.loc[(DF['month'] >= 10) | (DF['month'] <= 3)].copy()[Cols].mean(axis='index'),
    }
)
DF_AVG['AVG_Stg_winter_m_summer'] = DF_AVG['AVG_Stg_winter'] - DF_AVG['AVG_Stg_summer']

# %% Calc XY
DF_AVG[['L', 'R', 'C']] = DF_AVG.index.to_series().str.extract(r'L(\d+)_R(\d+)_C(\d+)').astype(int)
DF_AVG = DF_AVG.ws.Calc_XY(MB.Xmin, MB.Ymax, MB.cellsize)

# %% Convert to DA
DA = DF_AVG.set_index(['y', 'x'])[['AVG_Stg_summer', 'AVG_Stg_winter', 'AVG_Stg_winter_m_summer']].to_xarray()

# %% Save stage .IDFs
import imod

(MB.Pa.PoP_Out_MdlN / 'SFR/Stg').mkdir(parents=True, exist_ok=True)
for Par in DA.data_vars:
    DA_ = DA[Par].rio.write_crs('EPSG:28992')
    DA_.rio.to_raster(MB.Pa.PoP_Out_MdlN / f'SFR/Stg/SFR_{Par}_{MB.MdlN}.TIF')

# %% Save SFR+RIV stage .IDFs
# We need to do this cause the SFR only has values inside the catchment.
# ## Load RIV
import xarray as xra
from WS_Mdl.xr.spatial import clip_Mdl_area

RIV_Stg_summer = clip_Mdl_area(imod.idf.open(M.Pa.Mdl / r'In\RIV\NBr49\RIV_Stg_main_summer_NBr49.IDF'), M.MdlN)
RIV_Stg_winter = clip_Mdl_area(imod.idf.open(M.Pa.Mdl / r'In\RIV\NBr49\RIV_Stg_main_winter_NBr49.IDF'), M.MdlN)

# %% Align Coordinates, Join, Save
RIV_Stg_winter, SFR_Stg_winter = xra.align(RIV_Stg_winter, DA['AVG_Stg_winter'], join='outer')
RIV_Stg_summer, SFR_Stg_summer = xra.align(RIV_Stg_summer, DA['AVG_Stg_summer'], join='outer')

imod.idf.save(
    M.Pa.In / f'RIV/{MdlN}/RIV_Stg_main_winter_{M.MdlN}.IDF',
    xra.where(SFR_Stg_winter.notnull(), SFR_Stg_winter, RIV_Stg_winter),
)
imod.idf.save(
    M.Pa.In / f'RIV/{MdlN}/RIV_Stg_main_summer_{M.MdlN}.IDF',
    xra.where(SFR_Stg_summer.notnull(), SFR_Stg_summer, RIV_Stg_summer),
)
