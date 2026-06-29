import imod
import numpy as np
import pandas as pd
import xarray as xra
from WS_Mdl.core import *
from WS_Mdl.imod.mf6.read import MF6_block_to_DF

# %% Options
MdlN = "NBr101"
MdlN_B = "NBr68"

# %%
M = Mdl_N(MdlN)
MB = Mdl_N(MdlN_B)

# %% [markdown]
# # Load + process DF

# %%
Pa_SFR_Out = MB.Pa.Sim_In.parent / f"{MB.MdlN}.SFR6.stage.bin"

DF_ID = (
    MF6_block_to_DF(MB.Pa.SFR, "PACKAGEDATA")
    .loc[:, ["rno", "k", "i", "j"]]
    .sort_values("rno")
)

nrow = MB.N_R
ncol = MB.N_C
SFR_indices = (
    (DF_ID["k"].to_numpy() - 1) * nrow * ncol
    + (DF_ID["i"].to_numpy() - 1) * ncol
    + (DF_ID["j"].to_numpy() - 1)
)

# %%
SDate = np.datetime64(pd.to_datetime(str(MB.INI.SDATE), format="%Y%m%d"))
SFR_Stg = imod.mf6.open_dvs(
    Pa_SFR_Out,
    MB.Pa.GRB,
    indices=SFR_indices,
    simulation_start_time=SDate,
    time_unit="d",
)
SFR_Stg = SFR_Stg.where(SFR_Stg > -1.0e29)

# %%
DS_Stg = xra.Dataset(
    {
        "AVG_Stg_summer": SFR_Stg.where(
            (SFR_Stg["time"].dt.month > 3) & (SFR_Stg["time"].dt.month < 10),
            drop=True,
        ).mean("time"),
        "AVG_Stg_winter": SFR_Stg.where(
            (SFR_Stg["time"].dt.month >= 10) | (SFR_Stg["time"].dt.month <= 3),
            drop=True,
        ).mean("time"),
    }
)
DS_Stg["AVG_Stg_winter_m_summer"] = DS_Stg["AVG_Stg_winter"] - DS_Stg["AVG_Stg_summer"]

# %% [markdown]
# # Plot params

# %%
Par = "AVG_Stg_winter"
DS_Stg[Par].max("layer").plot.imshow()

# %%
DA = DS_Stg.to_array("Par")

# %%
DA.max("layer").plot.imshow(col="Par", col_wrap=3)

# %% [markdown]
# # Save stage .IDFs

# %%
for Par in ["AVG_Stg_summer", "AVG_Stg_winter", "AVG_Stg_winter_m_summer"]:
    DA_ = DS_Stg[Par].max("layer").rio.write_crs("EPSG:28992")
    DA_.rio.to_raster(MB.Pa.PoP_Out_MdlN / f"SFR/Stg/SFR_{Par}_{MB.MdlN}.TIF")

# %% [markdown]
# # Save SFR+RIV stage .IDFs
# We need to do this cause the SFR only has values inside the catchment.

# %% [markdown]
# ## Load RIV

from WS_Mdl.xr.spatial import clip_Mdl_area

# %%
RIV_Stg_summer = clip_Mdl_area(
    imod.idf.open(M.Pa.Mdl / r"In\RIV\NBr49\RIV_Stg_main_summer_NBr49.IDF"), M.MdlN
)
RIV_Stg_winter = clip_Mdl_area(
    imod.idf.open(M.Pa.Mdl / r"In\RIV\NBr49\RIV_Stg_main_winter_NBr49.IDF"), M.MdlN
)

# %% [markdown]
# # Align Coordinates, Join, Save

# %%
RIV_Stg_winter, SFR_Stg_winter = xra.align(
    RIV_Stg_winter, DS_Stg["AVG_Stg_winter"].max("layer"), join="outer"
)
RIV_Stg_summer, SFR_Stg_summer = xra.align(
    RIV_Stg_summer, DS_Stg["AVG_Stg_summer"].max("layer"), join="outer"
)

# %%
imod.idf.save(
    M.Pa.In / f"RIV/{MdlN}/RIV_Stg_main_winter_{M.MdlN}.IDF",
    xra.where(SFR_Stg_winter.notnull(), SFR_Stg_winter, RIV_Stg_winter),
)
imod.idf.save(
    M.Pa.In / f"RIV/{MdlN}/RIV_Stg_main_summer_{M.MdlN}.IDF",
    xra.where(SFR_Stg_summer.notnull(), SFR_Stg_summer, RIV_Stg_summer),
)
