"""Copied and edited ./RIV_set_null_where_SFR_NBr54.ipynb"""

# %% 0. Imports
from pathlib import Path

import imod
import numpy as np
import rioxarray as rxr
from WS_Mdl.core import *
from WS_Mdl.imod.xr import clip_Mdl_area

# %% 1. Options
MdlN = 'NBr146'
Pa_IDF_detail = r'g:\models\NBr\In\RIV\RIV_Cond_DETAILWATERGANGEN_NBr1.IDF'
Pa_IDF_main_RIV = r'g:\models\NBr\In\RIV\RIV_Cond_RIV_NBr1.IDF'
Pa_IDF_main_DRN = r'g:\models\NBr\In\RIV\RIV_Cond_DRN_NBr1.IDF'
Pa_SFR = r'G:\models\NBr\PoP\In\SFR\NBr143\SFR_rtp_NBr143.tif'
# Pa_boundary = r'g:\models\NBr\PoP\common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp'

# %% Loop
for Pa_IDF in [Pa_IDF_detail, Pa_IDF_main_RIV, Pa_IDF_main_DRN]:
    Pa_Out = Path(rf'g:\models\NBr\In\RIV\{MdlN}') / (f'{Path(Pa_IDF).stem.replace("NBr1", MdlN)}.idf')
    print(Pa_Out)
    # 2. Load IDF and boundary
    DA_init = imod.idf.open(Pa_IDF)
    # clip_Mdl_area(DA_init, MdlN).plot()

    # read polygon
    SFR = rxr.open_rasterio(Pa_SFR)
    SFR_full = SFR.reindex_like(DA_init)

    # apply mask → inside polygon becomes NaN
    DA = DA_init.where(SFR_full.isnull(), other=np.nan)
    # clip_Mdl_area(DA, MdlN).plot()

    # # %% zoom in to check
    # clip_Mdl_area(DA_init, MdlN).isel(x=slice(120, 130), y=slice(120, 130)).plot()

    # # %%
    # clip_Mdl_area(SFR, MdlN).isel(x=slice(120, 130), y=slice(120, 130)).plot()

    # # %%
    # clip_Mdl_area(DA, MdlN).isel(x=slice(120, 130), y=slice(120, 130)).plot()

    #  save as IDF
    Path(Pa_Out).parent.mkdir(parents=True, exist_ok=True)
    imod.idf.write(
        Path(Pa_Out).with_suffix('.idf'),
        clip_Mdl_area(DA, MdlN).squeeze(),
        nodata=-999.9900,
    )
