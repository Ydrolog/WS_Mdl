# %% Imports
import matplotlib.pyplot as plt  # Noqa: F401
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, green, sprint  # Noqa: F401
from WS_Mdl.imod.pop.sfr import SFR_CBC_to_DS

MdlN = 'NBr103'
Par = 'all'  # Can be a single parameter name, or a list of parameter names, or 'all' to export all parameters.


# %% Read CBC
sprint(Sep)
sprint('----- Exporting SFR CBC parameters to TIFs -----', style=green)
sprint(f'--- Reading {MdlN} CBC DS ... ', end='', set_time=True, verbose_out=False)
M = Mdl_N(MdlN)

DS = SFR_CBC_to_DS(MdlN)

if isinstance(Par, str):
    if Par.lower() == 'all':
        l_Par = list(DS.data_vars)
    else:
        if Par not in DS.data_vars:
            sprint('🔴', print_time=True)
            raise ValueError(f"'{Par}' not in DS data vars: {list(DS.data_vars)}. Cannot proceed.")
        l_Par = [Par]
else:
    l_Par = list(Par)
    for Par in l_Par:
        if Par not in DS.data_vars:
            sprint('🔴', print_time=True)
            raise ValueError(f"'{Par}' (in {l_Par}) not in DS data vars: {list(DS.data_vars)}. Cannot proceed.")

summer = (3 < DS.time.dt.month) & (DS.time.dt.month < 10)
winter = ~summer
sprint('🟢', print_time=True, verbose_in=True)

# %%
sprint('--- Exporting SFR CBC parameters to TIFs ... ', set_time2=True)
for V in DS.data_vars:
    if (
        DS[V].isel(time=0, layer=0).min().values != DS[V].isel(time=0, layer=0).max().values
    ):  # Assumes As with no meaningful values are constant across time and layer (min==0.0==max)
        sprint(f'  - Exporting {V: >20} to TIF ... ', end='', set_time=True)
        Par = V.split('_')[0]
        Pa_Out = M.Pa.PoP_Out_MdlN / f'SFR/{Par}'
        Pa_Out.mkdir(parents=True, exist_ok=True)

        # DS[V].sel(time=summer).mean(dim=['time', 'layer']).rio.to_raster(Pa_Out / f'{Par}_summer_AVG.tif')
        # DS[V].sel(time=winter).mean(dim=['time', 'layer']).rio.to_raster(Pa_Out / f'{Par}_winter_AVG.tif')
        # DS[V].mean(dim=['time', 'layer']).rio.to_raster(Pa_Out / f'{Par}_winter.tif')

        # 666 quantiles can be added

        sprint('🟢', print_time=True)

sprint('--- 🟢🟢🟢', print_time2=True)
sprint(Sep)

# %%
