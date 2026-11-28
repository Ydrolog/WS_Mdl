# %% Imports
import imod

# from WS_Mdl.imod.pop.sfr import SFR_Bin_to_DS
import rioxarray as rxr
from WS_Mdl.core import *

# %% Prep
MdlN = 'NBr147'
MdlN_B = 'NBr144'
M = Mdl_N(MdlN)
MB = Mdl_N(MdlN_B)  # B just for stages.

# %% Load DS
DS_winter = rxr.open_rasterio(MB.Pa.PoP_Out_MdlN / f'SFR/stage/stage_winter_AVG_{MdlN_B}.TIF', masked=True).squeeze()
DS_summer = rxr.open_rasterio(MB.Pa.PoP_Out_MdlN / f'SFR/stage/stage_summer_AVG_{MdlN_B}.TIF', masked=True).squeeze()

# %% Save SFR+RIV stage .IDFs
# We need to do this cause the SFR only has values inside the catchment.
# ## Load RIV
import xarray as xra
from WS_Mdl.xr.spatial import clip_Mdl_area

RIV_Stg_summer = clip_Mdl_area(imod.idf.open(M.Pa.Mdl / r'In\RIV\NBr49\RIV_Stg_main_summer_NBr49.IDF'), M.MdlN)
RIV_Stg_winter = clip_Mdl_area(imod.idf.open(M.Pa.Mdl / r'In\RIV\NBr49\RIV_Stg_main_winter_NBr49.IDF'), M.MdlN)

# %% Align Coordinates, Join, Save
RIV_Stg_winter, SFR_Stg_winter = xra.align(RIV_Stg_winter, DS_winter, join='outer')
RIV_Stg_summer, SFR_Stg_summer = xra.align(RIV_Stg_summer, DS_summer, join='outer')

imod.idf.save(
    M.Pa.In / f'RIV/{MdlN}/RIV_Stg_main_winter_{M.MdlN}.IDF',
    xra.where(SFR_Stg_winter.notnull(), SFR_Stg_winter, RIV_Stg_winter),
)
imod.idf.save(
    M.Pa.In / f'RIV/{MdlN}/RIV_Stg_main_summer_{M.MdlN}.IDF',
    xra.where(SFR_Stg_summer.notnull(), SFR_Stg_summer, RIV_Stg_summer),
)
