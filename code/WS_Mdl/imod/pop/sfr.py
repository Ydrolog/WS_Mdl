# %% Default imports
# import importlib as IL
import numpy as np
import pandas as pd
import rioxarray  # Noqa: F401
import xarray as xra
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import Sep, green, set_verbose, sprint

# %% End of default imports

__all__ = ['c_Stg_AVGs']


def c_Stg_Pctl(MdlN: str, Pctls: list = [5, 10, 50, 90, 95]):  # 666 Finish it off and add to Smk procedure
    # %% Prep
    M = Mdl_N(MdlN)

    # %% Load + Prep DF
    Pa_SFR_Out = M.Pa.Sim_In / f'{M.MdlN}.SFR6.obs.output.csv'
    DF_ = pd.read_csv(Pa_SFR_Out)

    date = M.SP_1st_DT + pd.to_timedelta(DF_['time'] - 1, unit='D')

    DF = pd.concat(
        [
            pd.DataFrame({'date': date, 'month': date.dt.month}, index=DF_.index),
            DF_.filter(like='STAGE_L'),
        ],
        axis=1,
    )

    # %% Calc Pctls
    # Cols = [c for c in DF.columns if c.startswith('STAGE_L')]
    # DF_AVG = pd.DataFrame(
    #     {
    #         'AVG_Stg_summer': DF.loc[(DF['month'] > 3) & (DF['month'] < 10)].copy()[Cols].mean(axis='index'),
    #         'AVG_Stg_winter': DF.loc[(DF['month'] >= 10) | (DF['month'] <= 3)].copy()[Cols].mean(axis='index'),
    #     }
    # )
    # DF_AVG['AVG_Stg_winter_m_summer'] = DF_AVG['AVG_Stg_winter'] - DF_AVG['AVG_Stg_summer']

    # # %% Calc XY
    # DF_AVG[['L', 'R', 'C']] = DF_AVG.index.to_series().str.extract(r'L(\d+)_R(\d+)_C(\d+)').astype(int)
    # DF_AVG = DF_AVG.ws.Calc_XY(M.Xmin, M.Ymax, M.cellsize)

    # # %% Convert to DA
    # DA = DF_AVG.set_index(['y', 'x'])[['AVG_Stg_summer', 'AVG_Stg_winter', 'AVG_Stg_winter_m_summer']].to_xarray()

    # # %% Save stage .IDFs
    # (M.Pa.PoP_Out_MdlN / 'SFR/Stg').mkdir(parents=True, exist_ok=True)
    # for Par in DA.data_vars:
    #     DA_ = DA[Par].rio.write_crs('EPSG:28992')
    #     DA_.rio.to_raster(M.Pa.PoP_Out_MdlN / f'SFR/Stg/SFR_{Par}_{M.MdlN}.TIF')

    # return DA


def c_Stg_AVGs(MdlN, start_year: str = 'from_INI', end_year: str = 'from_INI'):
    """
    Calculated Stg and depth AVGs and saves them as TIFs.
    """
    sprint(Sep)
    sprint(f'----- SFR stage & depth AVGs - {MdlN} -----\n', style=green)
    # Imports
    sprint('--- Importing...', end='', set_time=True)
    from WS_Mdl.imod.sfr.export import Par_to_Rst

    sprint('🟢', print_time=True)

    # %% Prep
    sprint('--- Loading SFR Out to DF...', end='')
    M = Mdl_N(MdlN)
    start_year = M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)

    # %% Load + Prep DF
    Pa_SFR_Out = M.Pa.Sim_In / f'{M.MdlN}.SFR6.obs.output.csv'  # 666 Need to move to Bin at some point
    DF_ = pd.read_csv(Pa_SFR_Out)

    date = M.SP_1st_DT + pd.to_timedelta(DF_['time'] - 1, unit='D')

    DF = pd.concat(
        [
            pd.DataFrame({'date': date, 'month': date.dt.month}, index=DF_.index),
            DF_.filter(like='STAGE_L'),
        ],
        axis=1,
    )
    DF = DF.loc[(DF.date.dt.year <= end_year) & (DF.date.dt.year >= start_year)]
    sprint('🟢', print_time=True)

    # %% Calc AVGs
    sprint('--- Calculating AVGs & converting to DA...', end='', set_time=True)
    Cols = [c for c in DF.columns if c.startswith('STAGE_L')]
    DF_AVG = pd.DataFrame(
        {
            'Stg_summer_AVG': DF.loc[(DF['month'] > 3) & (DF['month'] < 10)].copy()[Cols].mean(axis='index'),
            'Stg_winter_AVG': DF.loc[(DF['month'] >= 10) | (DF['month'] <= 3)].copy()[Cols].mean(axis='index'),
            'Stg_AVG': DF.copy()[Cols].mean(axis='index'),
        }
    )
    DF_AVG['Stg_winter_m_summer_AVG'] = DF_AVG['Stg_winter_AVG'] - DF_AVG['Stg_summer_AVG']

    # %% Calc XY
    DF_AVG[['L', 'R', 'C']] = DF_AVG.index.to_series().str.extract(r'L(\d+)_R(\d+)_C(\d+)').astype(int)
    DF_AVG = DF_AVG.ws.Calc_XY(M.Xmin, M.Ymax, M.cellsize)

    # %% Convert to DA
    DA = DF_AVG.set_index(['y', 'x'])[
        ['Stg_summer_AVG', 'Stg_winter_AVG', 'Stg_winter_m_summer_AVG', 'Stg_AVG']
    ].to_xarray()
    sprint('🟢', print_time=True)

    # %% Load rtp (for depth Calcs)
    sprint('--- Loading rtp...', end='', set_time=True)
    A_rtp = Par_to_Rst(M.MdlN, 'rtp', verbose=False)
    set_verbose(True)
    DA_rtp = xra.DataArray(
        A_rtp,
        dims=('y', 'x'),
        coords={
            'x': M.Xmin + (np.arange(A_rtp.shape[1]) + 0.5) * M.cellsize,
            'y': M.Ymax - (np.arange(A_rtp.shape[0]) + 0.5) * M.cellsize,
        },
    ).rio.write_crs('EPSG:28992')
    sprint('🟢', print_time=True)

    # %% Save stage TIFs, then depth TIFs
    sprint('--- Saving stage TIFs...', set_time=True)
    (M.Pa.PoP_Out_MdlN / 'SFR/Stg').mkdir(parents=True, exist_ok=True)
    (M.Pa.PoP_Out_MdlN / 'SFR/depth').mkdir(parents=True, exist_ok=True)
    for Par in DA.data_vars:
        print(f'  - {Par}...', end='', flush=True)
        DA_ = DA[Par].rio.write_crs('EPSG:28992')
        DA_.rio.to_raster(M.Pa.PoP_Out_MdlN / f'SFR/Stg/SFR_{Par}_{M.MdlN}.TIF')
        if '_m_' not in Par:  # depth winter_m_summer is the same as Stg, thus unnecessary.
            DA_, DA_rtp_ = xra.align(DA_, DA_rtp, join='left')
            DA_depth = DA_ - DA_rtp_
            DA_depth.name = Par.replace('Stg', 'depth')
            DA_depth.rio.to_raster(M.Pa.PoP_Out_MdlN / f'SFR/depth/SFR_{DA_depth.name}_{M.MdlN}.TIF')
        sprint('🟢', print_time=True)

    sprint(Sep)

    # 666 Change function to 1st convert whole DF into DA (not just AVGs) and return that DA so other things (e.g. AVG stage per year) can be calculated from that DA.


# %%
def stage_TS(
    MdlN: str,
    MdlN_RIV: str,
    N_system_RIV: int = None,
    N_system_DRN: int = None,
    min_date: str = None,
    max_date: str = None,
    load_HD: bool = True,
    load_HD_RIV: bool = True,
    load_P: bool = True,
    SFR_Out_Col_prefix: str = 'STAGE_',
):
    """
    Generates time series plots of SFR stage data from model output.
    """

    # %% Inputs for testing
    MdlN = 'NBr100'
    MdlN_RIV = 'NBr102'
    N_system_RIV: int = 3
    N_system_DRN: int = 1
    min_date = None
    max_date = '2001-12-31'
    load_HD: bool = True
    load_HD_RIV: bool = True
    load_P: bool = True
    SFR_Out_Col_prefix: str = 'STAGE_'

    # %% Extra imports
    sprint(Sep)
    sprint(
        f'----- Generating SFR stage time series plots for model: {MdlN} and RIV model: {MdlN_RIV}, system: {N_system_RIV} -----\n'
    )
    sprint('--- Extra imports...', end='', set_time=True)
    import re
    from datetime import datetime as DT

    import plotly.graph_objects as go
    from imod import idf
    from tqdm.dask import TqdmCallback
    from WS_Mdl.imod import ini, prj
    from WS_Mdl.imod.mf6.obs import o_HD_OBS_L_Bin
    from WS_Mdl.imod.msw.mete_grid import to_DF
    from WS_Mdl.imod.msw.meteo import to_XA
    from WS_Mdl.imod.prj import r_with_OBS
    from WS_Mdl.imod.sfr.info import SFR_PkgD_to_DF, get_SFR_OBS_Out_Pas
    from WS_Mdl.imod.xr import clip_Mdl_area
    from WS_Mdl.viz.ts import SFR_reach_TS
    from WS_Mdl.xr.spatial import get_value

    sprint('🟢', print_time=True)

    # region ----- Load data -------------------------------------------------------
    sprint('--- Loading data ... ')

    # %% Load PRJ data
    sprint(' -- Loading PRJ data ... ', end='', set_time=True)
    M = Mdl_N(MdlN)
    M_RIV = Mdl_N(MdlN_RIV)
    Pa = M.Pa
    Pa_RIV = M_RIV.Pa

    d_INI = ini.as_d(Pa.INI)
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = ini.Mdl_Dmns(Pa.INI)
    SP_date_1st = DT.strftime(DT.strptime(d_INI['SDATE'], '%Y%m%d'), '%Y-%m-%d')
    dx = dy = float(d_INI['CELLSIZE'])

    d_INI_RIV = ini.as_d(Pa_RIV.INI)
    SP_date_1st_RIV = DT.strftime(DT.strptime(d_INI_RIV['SDATE'], '%Y%m%d'), '%Y-%m-%d')

    # Check that model dimensions are the same
    if (Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C) != ini.Mdl_Dmns(Pa_RIV.INI):
        print('Warning: Model dimensions for RIV model differ from main model.')

    PRJ, PRJ_OBS = r_with_OBS(Pa.PRJ)
    sprint('🟢', print_time=True)

    # %% Load SFR data
    sprint(' -- Loading SFR data ... ', end='', set_time=True, verbose_out=False)
    DF_ = pd.read_csv(get_SFR_OBS_Out_Pas(MdlN))  # 666 replace 1. This needs to be standardized.
    DF = DF_[[i for i in DF_.columns if i == 'time' or 'L' in i]].copy()
    DF['time'] = pd.to_datetime(SP_date_1st) + pd.to_timedelta(DF['time'] - 1, unit='D')
    if min_date is not None:
        DF = DF.loc[DF['time'] >= pd.to_datetime(min_date)]
    if max_date is not None:
        DF = DF.loc[DF['time'] <= pd.to_datetime(max_date)]
    DF = DF.reset_index(drop=True)

    GDF_SFR = SFR_PkgD_to_DF(MdlN)
    reach_col = 'rno' if 'rno' in GDF_SFR.columns else 'reach'
    sfr_by_reach = GDF_SFR.set_index(reach_col, drop=False)
    sfr_by_cell = GDF_SFR.set_index(['k', 'i', 'j'], drop=False)

    def _first_sfr_row(row):
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    sprint('🟢', print_time=True, verbose_in=True)

    # %% Load RIV data
    sprint(' -- Loading RIV data...', end='', set_time=True, verbose_out=False)
    RIV_params = ['conductance', 'stage', 'bottom_elevation', 'infiltration_factor']
    PRJ_RIV, PRJ_OBS_RIV = prj.r_with_OBS(Pa_RIV.PRJ)

    l_N_system_RIV_print = []
    for i in range(PRJ_RIV['(riv)']['n_system']):
        l_N_system_RIV_print.append(f'System {i + 1}:')
        for j in RIV_params:
            if 'path' in PRJ_RIV['(riv)'][j][i]:
                l_N_system_RIV_print.append(f'\t{j:<20}: {PRJ_RIV["(riv)"][j][i]["path"].name}')
            elif 'constant' in PRJ_RIV['(riv)'][j][i]:
                l_N_system_RIV_print.append(f'\t{j:<20}: {PRJ_RIV["(riv)"][j][i]["constant"]}')
            else:
                l_N_system_RIV_print.append(f'\t{j:<20}: N/A')

    str_N_system_RIV_print = '\n'.join(l_N_system_RIV_print)

    if N_system_RIV is None:
        print(
            f'  - You need to choose one of {PRJ_RIV["(riv)"]["n_system"]} river systems.\nHere is some information about the RIV systems:\n{str_N_system_RIV_print}\n'
        )
        N_system_RIV = int(input('Select the number of the RIV system you want to plot (1-indexed).'))
    elif N_system_RIV < 1 or N_system_RIV > PRJ_RIV['(riv)']['n_system']:
        print(f'Invalid system number. It should be >= 1 & <= {PRJ_RIV["(riv)"]["n_system"]}.')
        # return

    A_RIV_Stg_winter = clip_Mdl_area(idf.open(PRJ_RIV['(riv)']['stage'][N_system_RIV - 1]['path']), MdlN=MdlN)
    A_RIV_Stg_summer = clip_Mdl_area(
        idf.open(PRJ_RIV['(riv)']['stage'][PRJ_RIV['(riv)']['n_system'] + N_system_RIV - 1]['path']), MdlN=MdlN
    )
    # A_RIV_Btm = clip_Mdl_area(idf.open(PRJ_RIV['(riv)']['bottom_elevation'][N_system_RIV - 1]['path']), MdlN=MdlN)

    sprint('🟢', print_time=True, verbose_in=True)
    # if A_RIV_Btm.notnull().sum().values == (A_RIV_Btm == A_RIV_Stg_winter).sum().values:
    #     sprint('\tAll river bottom elevations are equal to stage elevations.')

    # %% Load DRN data
    sprint(' -- Loading DRN data...', end='', set_time=True, verbose_out=False)
    DRN_params = ['conductance', 'elevation', 'n_system', 'active']
    l_N_system_DRN_print = []
    for i in range(PRJ['(drn)']['n_system']):
        l_N_system_DRN_print.append(f'System {i + 1}:')
        if i in PRJ['(drn)'][i]:
            for j in DRN_params:
                if 'path' in PRJ['(drn)'][j][i]:
                    l_N_system_DRN_print.append(f'\t{j:<20}: {PRJ["(drn)"][j][i]["path"].name}')
                elif 'constant' in PRJ['(drn)'][j][i]:
                    l_N_system_DRN_print.append(f'\t{j:<20}: {PRJ["(drn)"][j][i]["constant"]}')
                else:
                    l_N_system_DRN_print.append(f'\t{j:<20}: N/A')

    str_N_system_DRN_print = '\n'.join(l_N_system_DRN_print)

    if N_system_DRN is None:
        print(
            f'  - You need to choose one of {PRJ["(drn)"]["n_system"]} river systems.\nHere is some information about the DRN systems:\n{str_N_system_DRN_print}\n'
        )
        N_system_DRN = int(input('Select the number of the DRN system you want to plot (1-indexed).'))
    elif N_system_DRN < 1 or N_system_DRN > PRJ['(drn)']['n_system']:
        print(f'Invalid system number. It should be >= 1 & <= {PRJ["(drn)"]["n_system"]}.')
        # return

    A_DRN_Elv = clip_Mdl_area(idf.open(PRJ['(drn)']['elevation'][N_system_DRN - 1]['path']), MdlN=MdlN)
    sprint('🟢', print_time=True, verbose_in=True)

    # %% Load TOP and BOT data
    sprint(' -- Loading TOP BOT data...', end='', set_time=True, verbose_out=False)
    l_Pa_TOP = [i['path'] for i in PRJ['(top)']['top']]
    A_TOP = clip_Mdl_area(
        idf.open(l_Pa_TOP, pattern=r'TOP_L{layer}_{name}'), MdlN=MdlN
    )  # We're just doing this to avoid errors - using {name} to capture the model number part - idf will use it for the DataArray name.
    l_Pa_BOT = [i['path'] for i in PRJ['(bot)']['bottom']]
    A_BOT = clip_Mdl_area(idf.open(l_Pa_BOT, pattern=r'BOT_L{layer}_{name}'), MdlN=MdlN)

    # Get layers relevant for SFR (HDS output can be too big to load to memory, and it's also efficient to just save the relevant layers)
    DF_ = pd.DataFrame({'L': GDF_SFR.k.value_counts().index, 'count': GDF_SFR.k.value_counts()})
    DF_['percentage'] = (GDF_SFR.k.value_counts(normalize=True) * 100).apply(lambda x: round(x, 2))
    l_SFR_Ls = [int(i) for i in sorted(DF_.loc[DF_['percentage'] >= 1, 'L'].unique())]
    sprint(
        f'  - Only layers containing >=1% of SFR reaches were loaded: {l_SFR_Ls}.\n\t You can still request for a TS from the other layers, but it may take a while to load the data if you do.'
    )
    sprint('🟢', print_time=True, verbose_in=True)

    # %% meteo
    DF_meteo = timed_Exe(to_DF, PRJ, pre=' -- Listing P files in a DF... ', post='🟢')

    set_verbose(True)

    # %% Load HD data
    sprint(f' -- Reading {MdlN} HD data ... ', end='', set_time=True)
    if load_HD:
        try:
            A_HD_ = o_HD_OBS_L_Bin(
                MdlN,
                l_L=l_SFR_Ls,
                min_date=DF['time'].min(),
                max_date=DF['time'].max(),
            ).astype('float32')
        except Exception as e:
            print(f'🔴🔴🔴 - An error occurred while loading HD data: {e}')
    sprint('🟢', print_time=True, verbose_in=True)

    sprint(f' -- Reading {MdlN_RIV} HD data ... ', end='', set_time=True)
    if load_HD_RIV:
        try:
            A_HD_RIV_ = o_HD_OBS_L_Bin(
                MdlN_RIV,
                l_L=l_SFR_Ls,
                min_date=DF['time'].min(),
                max_date=DF['time'].max(),
            ).astype('float32')
        except Exception as e:
            print(f'🔴🔴🔴 - An error occurred while loading HD data: {e}')
    sprint('🟢', print_time=True, verbose_in=True)
    sprint()
    # endregion ---------- Load data ----------

    # %% Iteration loop
    while True:
        In1 = input(
            f'Start date is {DF["time"].min()} to {DF["time"].max()} for model {MdlN}.\n\nPress any key except Y and E to continue using this temporal extent.\nPress Y to set another temporal extent.\nPress E to exit.\n'
        )
        if In1.upper() == 'E':
            break

        start_date = (
            pd.to_datetime(input('Enter start date (YYYY-MM-DD):\n')) if In1.upper() == 'Y' else DF['time'].min()
        )
        end_date = pd.to_datetime(input('Enter end date (YYYY-MM-DD):\n')) if In1.upper() == 'Y' else DF['time'].max()
        DF_trim = (
            DF.copy().loc[(DF['time'] >= start_date) & (DF['time'] <= end_date)].reset_index(drop=True)
            if In1.upper() == 'Y'
            else DF.copy()
        )
        X_axis = DF_trim['time'].to_numpy(copy=False)
        month = DF_trim['time'].dt.month.to_numpy(copy=False)
        n_times = len(DF_trim)

        if load_P:
            DF_meteo_DT_trim = DF_meteo.loc[(DF_meteo['DT'] >= start_date) & (DF_meteo['DT'] <= end_date)].copy()
            A_P = to_XA(DF_meteo_DT_trim, 'P', MdlN, clip=True)

        with TqdmCallback(desc=' -- Loading HD data for SFR-relevant layers and specified time range ...'):
            if load_HD:
                l_union = [i for i in l_SFR_Ls if i in A_HD_.layer.values]
                A_HD = A_HD_.sel(layer=l_union).sel(time=slice(start_date, end_date)).compute()
            if load_HD_RIV:
                l_union = [i for i in l_SFR_Ls if i in A_HD_RIV_.layer.values]
                A_HD_RIV = A_HD_RIV_.sel(layer=l_union).sel(time=slice(start_date, end_date)).compute()
            sprint(' 🟢')

        while True:
            try:
                In2 = input(
                    "Provide a cell ID (L R C) (with spaces or commas as separators) or a reach number. If you're providing a reach number, prefix it with 'R' (e.g., R15). Type 'E' to quit:\n"
                )

                # IL.reload(plot_SFR) # Relic. For development so it doesn't need to reload big files.
                # Re-bind the function from the reloaded module
                plot_SFR_reach_TS = SFR_reach_TS

                sprint('  - Loading data for specified reach/cell...')

                if In2.upper() == 'E':
                    break

                elif In2.upper().startswith('R'):
                    reach = int(In2.upper().replace('R', ''))
                    sfr_row = _first_sfr_row(sfr_by_reach.loc[reach])
                    L, R, C = int(sfr_row.k), int(sfr_row.i), int(sfr_row.j)
                else:
                    parts = re.split(r'[,\s]+', In2.strip())  # Split by commas and/or whitespace
                    L, R, C = [int(j) for j in parts]
                    sfr_row = _first_sfr_row(sfr_by_cell.loc[(L, R, C)])
                    reach = int(sfr_row[reach_col])
                X, Y = float(sfr_row.x), float(sfr_row.y)
                rtp = float(sfr_row.rtp)

                if load_P:
                    P_ts = get_value(A_P, X, Y, dx, dy)

                # Extract head time series at this location (compute to load from Dask)
                if load_HD:
                    try:
                        HD_ts = get_value(A_HD.sel(time=slice(start_date, end_date)), X, Y, dx, dy, L=L)
                    except Exception as e:
                        print(
                            f"L {L} probably contains less than 1% of the SFR reaches, so its HD wasn't loaded.\nAn error occurred while extracting head time series: {e}"
                        )
                        print('Attempting to load full head data for the specified layer...')
                        A_HD_L = A_HD_.sel(layer=L)
                        HD_ts = get_value(A_HD_L.sel(time=slice(start_date, end_date)), X, Y, dx, dy)
                    HD_values = HD_ts.values

                if load_HD_RIV:
                    try:
                        HD_ts_RIV = get_value(A_HD_RIV.sel(time=slice(start_date, end_date)), X, Y, dx, dy, L=L)
                    except Exception as e:
                        print(
                            f"L {L} probably contains less than 1% of the SFR reaches, so its HD wasn't loaded.\nAn error occurred while extracting head time series: {e}"
                        )
                        print('Attempting to load full head data for the specified layer...')
                        A_HD_L_RIV = A_HD_RIV_.sel(layer=L)
                        HD_ts_RIV = get_value(A_HD_L_RIV.sel(time=slice(start_date, end_date)), X, Y, dx, dy)
                    HD_RIV_values = HD_ts_RIV.values

                    # required keys: plot_type, y. Rest are kwargs for the plot type (e.g., line dict for scatter, marker dict for bar, etc.)

                # region - Design dictionary for plotting. Mandatory keys: plot_type, y. Rest are kwargs for the plot type (e.g., line dict for scatter, marker dict for bar, etc.)
                d_plot = {}

                load_P and d_plot.update(
                    {
                        f'{"Precipitation":<17}': {
                            'plot_type': go.Bar,
                            'y': P_ts.values if hasattr(P_ts, 'values') else P_ts,
                            'kwargs': {
                                'marker': dict(color='#919cd5', line=dict(color='#919cd5', width=3)),
                                'hovertemplate': '%{y:8.1f} mm',
                                'yaxis': 'y2',
                            },
                        }
                    }
                )

                load_HD and d_plot.update(
                    {
                        f'{"Head " + MdlN:<17}': {
                            'plot_type': go.Scatter,
                            'y': HD_values,
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#b63a3a', width=3),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                load_HD_RIV and d_plot.update(
                    {
                        f'{"Head " + MdlN_RIV:<17}': {
                            'plot_type': go.Scatter,
                            'y': HD_RIV_values,
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#3a7fb8', width=3),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"SFR stage":<17}': {
                            'plot_type': go.Scatter,
                            'y': DF_trim[f'{SFR_Out_Col_prefix}L{L}_R{R}_C{C}'].to_numpy(copy=False),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#ff0000', width=3),
                                'hovertemplate': '%{y:3.3f} mNAP',
                                # showlegend=True
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"SFR riverbet top":<17}': {
                            'plot_type': go.Scatter,
                            'y': np.full(n_times, rtp),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#ff0000', width=2, dash='dash'),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                RIV_Stg_winter = round(float(get_value(A_RIV_Stg_winter, X, Y, dx, dy)), 3)
                RIV_Stg_summer = round(float(get_value(A_RIV_Stg_summer, X, Y, dx, dy)), 3)
                RIV_Stg = np.where(np.isin(month, [10, 11, 12, 1, 2, 3]), RIV_Stg_winter, RIV_Stg_summer)

                d_plot.update(
                    {
                        f'{"RIV stage":<17}': {
                            'plot_type': go.Scatter,
                            'y': RIV_Stg,
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#0000ff', width=3),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )
                # d_plot.update(
                #     {
                #         f'{"RIV bottom":<17}': {
                #             'plot_type': go.Scatter,
                #             'y': [round(float(get_value(A_RIV_Btm, X, Y, dx, dy)), 3)] * len(DF_trim),
                #             'kwargs': {
                #                 'mode': 'lines',
                #                 'line': dict(color='#0000ff', width=2, dash='dash'),
                #                 'hovertemplate': '%{y:3.3f} mNAP',
                #             },
                #         }
                #     }
                # )
                d_plot.update(
                    {
                        f'{"DRN elevation":<17}': {
                            'plot_type': go.Scatter,
                            'y': np.full(n_times, round(float(get_value(A_DRN_Elv, X, Y, dx, dy)), 3)),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#d000ff', width=2, dash='dot'),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"top":<17}': {
                            'plot_type': go.Scatter,
                            'y': np.full(n_times, round(float(get_value(A_TOP, X, Y, dx, dy, L=L)), 3)),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#a47300', width=2, dash='dash'),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"bottom":<17}': {
                            'plot_type': go.Scatter,
                            'y': np.full(n_times, round(float(get_value(A_BOT, X, Y, dx, dy, L=L)), 3)),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#a47300', width=2, dash='dash'),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )
                # endregion

                r_info = f'reach: {reach}, L: {L}, R: {R}, C: {C}, X: {X}, Y: {Y}, MdlN: {MdlN}, MdlN_RIV: {MdlN_RIV}'

                sprint('  - Plotting...')

                Pa_Out = Pa.PoP_Out_MdlN / f'SFR/Stg/TS/r{reach}.html'
                Pa_Out.parent.mkdir(parents=True, exist_ok=True)

                plot_SFR_reach_TS(sub_title=r_info, X_axis=X_axis, d_plot=d_plot, Pa_Out=Pa_Out)
                sprint('  🟢🟢 - Success!')
            except Exception as e:  # probubly redundant to ploy that much
                err_lines = [f' 🔴🔴 - Error type: {type(e).__name__}', f' - Details: {e}']

                if isinstance(e, KeyError):
                    missing_key = str(e).strip('\'"')
                    err_lines.append(f' - Missing key/column: {missing_key}')

                    if re.match(r'^L\d+_R\d+_C\d+$', missing_key):
                        obs_cols = [
                            c
                            for c in DF_trim.columns
                            if isinstance(c, str) and c.startswith('L') and '_R' in c and '_C' in c
                        ]
                        err_lines.append(
                            ' - This usually means the SFR OBS CSV has no stage column for the selected cell (or a different naming/indexing convention is used).'
                        )
                        if len(obs_cols) > 0:
                            err_lines.append(
                                f' - Example available SFR columns: {", ".join(obs_cols[:5])}{" ..." if len(obs_cols) > 5 else ""}'
                            )
                        if 'reach' in locals():
                            err_lines.append(
                                f' - Requested selection resolved to reach {reach} at L{L}_R{R}_C{C}. Verify this cell exists in the OBS output and rerun the model if needed.'
                            )

                elif isinstance(e, ValueError):
                    err_lines.append(
                        " - Input format should be either: 'R<reach>' (e.g., R15) or three integers 'L R C' (e.g., 11 51 50)."
                    )

                sprint('\n'.join(err_lines))
                sprint('Please try again.')
    sprint('🟢🟢🟢 - Finished SFR stage TS plotting.')
    sprint(Sep)


# %%
