# %% General Imports
import datetime as DT

import pandas as pd
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import sprint


# %%
def Agg_outlet_TS(MdlN: str, Pa_clip: str, save: bool = True, overwrite: bool = False):

    M = Mdl_N(MdlN)

    # %% Create DF
    DF = pd.DataFrame({'date': pd.date_range(start=M.SP_1st_DT, end=M.SP_last_DT - DT.timedelta(days=1))})
    l_Fi = [i for i in M.Pa.Sim_Out.glob('*_OBS*.csv') if 'DRN' in i.name or 'RIV' in i.name]

    # %% RIV & DRN
    for F in l_Fi[:]:
        sprint(f'Aggregating {F.stem} ... ', end='', set_time=True)
        DF[F.stem] = pd.read_csv(F).copy().drop(columns=['time']).sum(axis=1) * (-1)
        sprint('🟢', print_time=True)

    # %% SFR
    Pa_SFR = next(iter(M.Pa.Sim_Out.glob('*SFR*.csv')), None)
    if Pa_SFR:
        DF['SFR_outlet'] = pd.read_csv(Pa_SFR)['OUTLET_DOWNSTREAM-FLOW'] * (-1)

    # MSW qrun
    import geopandas as gpd

    GDF_CB = gpd.read_file(Pa_clip)

    if (GDF_CB.crs != CRS) or (GDF_CB.crs is None):  # Set default CRS
        GDF_CB = GDF_CB.set_crs(CRS, allow_override=True)

    # qrun
    try:
        # Load qrun IDFs to xarray
        import imod

        A = imod.idf.open(M.Pa.MSW / 'bdgqrun/area_L1.IDF')  # Area array
        DA_qrun = imod.idf.open(M.Pa.MSW / 'bdgqrun/bdgqrun_*_L*.IDF')
        DA_qrun = DA_qrun * A * (-1)

        # Expose X/Y as spatial dimensions for rioxarray; this is needed for clipping.
        DA_Qrun_Rio = DA_qrun.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)

        # Ensure CRS compatibility # If CRS metadata is missing, use default
        if DA_Qrun_Rio.rio.crs != CRS:
            DA_Qrun_Rio.rio.write_crs(CRS, inplace=True)

        # Clip while preserving the original x/y grid shape; outside the polygon becomes NaN.
        DA_qrun_clip = DA_Qrun_Rio.rio.clip(GDF_CB.geometry, CRS, drop=False)

        # Sum and append
        DA_qrun_clip_sum = DA_qrun_clip.sum(dim=('layer', 'x', 'y'))
        DF['qrun'] = DA_qrun_clip_sum.to_dataframe(name='qrun').reset_index()['qrun']
        sprint(f'  🟢 - Sucessfully appended {MdlN} MSW qrun')
    except Exception as e:
        sprint(f'  🔴 - Failed to read {MdlN} MSW qrun. Error:\n{e}')

    if save:
        Pa_Out = M.Pa.PoP_Out_MdlN / f'outlet_TS_{MdlN}.csv'
        Pa_Out.parent.mkdir(parents=True, exist_ok=True)
        DF.to_csv(Pa_Out, index=False)

    return DF
