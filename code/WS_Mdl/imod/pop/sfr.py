# import importlib as IL
import re
from datetime import datetime as DT

import imod
import pandas as pd
import plotly.graph_objects as go
from imod.msw.mete_grid import to_DF
from tqdm.dask import TqdmCallback
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, set_verbose, sprint
from WS_Mdl.imod import ini, prj
from WS_Mdl.imod.msw.meteo import to_XA
from WS_Mdl.imod.prj import r_with_OBS
from WS_Mdl.imod.sfr.info import SFR_PkgD_to_DF, get_SFR_OBS_Out_Pas, reach_to_cell_id, reach_to_XY
from WS_Mdl.imod.xr import clip_Mdl_Aa
from WS_Mdl.viz.ts import SFR_reach_TS
from WS_Mdl.xr.spatial import get_value


def stage_TS(
    MdlN: str,
    MdlN_RIV: str,
    N_system_RIV: int = None,
    N_system_DRN: int = None,
    load_HD: bool = True,
    load_HD_RIV: bool = True,
    load_P: bool = True,
    SFR_Out_Col_prefix: str = 'STG_',
):
    """
    Generates time series plots of SFR stage data from model output.
    """

    sprint(Sep)
    sprint(
        f'----- Generating SFR stage time series plots for model: {MdlN} and RIV model: {MdlN_RIV}, system: {N_system_RIV} -----\n'
    )
    set_verbose(False)

    # region ----- Load data -------------------------------------------------------
    print('--- Loading data ... ')

    print(' -- Loading PRJ data ... ', end='')
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
    print('🟢')

    print(' -- Loading SFR data ... ', end='')
    DF_ = pd.read_csv(get_SFR_OBS_Out_Pas(MdlN))  # 666 replace 1. This needs to be standardized.
    DF = DF_[[i for i in DF_.columns if i == 'time' or 'L' in i]].copy()
    DF['time'] = pd.to_datetime(SP_date_1st) + pd.to_timedelta(DF['time'] - 1, unit='D')

    GDF_SFR = SFR_PkgD_to_DF(MdlN)
    print('🟢')

    print(' -- Loading RIV data...', end='')
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
        return

    A_RIV_Stg = clip_Mdl_Aa(imod.idf.open(PRJ_RIV['(riv)']['stage'][N_system_RIV - 1]['path']), MdlN=MdlN)
    A_RIV_Btm = clip_Mdl_Aa(imod.idf.open(PRJ_RIV['(riv)']['bottom_elevation'][N_system_RIV - 1]['path']), MdlN=MdlN)

    print('🟢')
    if A_RIV_Btm.notnull().sum().values == (A_RIV_Btm == A_RIV_Stg).sum().values:
        print('\tAll river bottom elevations are equal to stage elevations.')

    print(' -- Loading DRN data...', end='')
    DRN_params = ['conductance', 'elevation', 'n_system', 'active']
    l_N_system_DRN_print = []
    for i in range(PRJ['(drn)']['n_system']):
        l_N_system_DRN_print.append(f'System {i + 1}:')
        if i in PRJ['(drn)'][j]:
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
        return

    A_DRN_Elv = clip_Mdl_Aa(imod.idf.open(PRJ['(drn)']['elevation'][N_system_DRN - 1]['path']), MdlN=MdlN)
    print('🟢')

    print(' -- Loading TOP BOT data...', end='')
    l_Pa_TOP = [i['path'] for i in PRJ['(top)']['top']]
    A_TOP = clip_Mdl_Aa(
        imod.idf.open(l_Pa_TOP, pattern=r'TOP_L{layer}_{name}'), MdlN=MdlN
    )  # We're just doing this to avoid errors - using {name} to capture the model number part - imod will use it for the DataArray name.
    l_Pa_BOT = [i['path'] for i in PRJ['(bot)']['bottom']]
    A_BOT = clip_Mdl_Aa(imod.idf.open(l_Pa_BOT, pattern=r'BOT_L{layer}_{name}'), MdlN=MdlN)
    print('🟢')

    # Get layers relevant for SFR (HDS output can be too big to load to memory, and it's also efficient to just save the relevant layers)
    DF_ = pd.DataFrame({'L': GDF_SFR.k.value_counts().index, 'count': GDF_SFR.k.value_counts()})
    DF_['percentage'] = (GDF_SFR.k.value_counts(normalize=True) * 100).apply(lambda x: round(x, 2))
    l_SFR_Ls = [int(i) for i in sorted(DF_.loc[DF_['percentage'] >= 1, 'L'].unique())]
    print(' -- Reading HD data... ', end='')
    print('🟢')
    print(
        f'  - Only layers containing >=1% of SFR reaches were loaded: {l_SFR_Ls}.\n\t You can still request for a TS from the other layers, but it may take a while to load the data if you do.'
    )

    print(' -- Loading precipitation data ... ', end='')
    DF_meteo = to_DF(PRJ)
    print('🟢')

    set_verbose(True)

    print(' -- Reading HD data ... ', end='')
    if load_HD:
        try:
            A_HD_ = imod.mf6.open_hds(
                hds_path=Pa.HD_Out_Bin,
                grb_path=Pa.DIS_GRB,
                simulation_start_time=pd.to_datetime(SP_date_1st),
                time_unit='d',
            ).astype('float32')
        except Exception as e:
            print(f'🔴🔴🔴 - An error occurred while loading HD data: {e}')

    if load_HD_RIV:
        try:
            A_HD_RIV_ = imod.mf6.open_hds(
                hds_path=Pa_RIV.HD_Out_Bin,
                grb_path=Pa_RIV.DIS_GRB,
                simulation_start_time=pd.to_datetime(SP_date_1st_RIV),
                time_unit='d',
            ).astype('float32')
        except Exception as e:
            print(f'🔴🔴🔴 - An error occurred while loading HD data: {e}')
    print('🟢')
    sprint()
    # endregion ---------- Load data ----------

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

        if load_P:
            DF_meteo_DT_trim = DF_meteo.loc[(DF_meteo['DT'] >= start_date) & (DF_meteo['DT'] <= end_date)].copy()
            A_P = to_XA(DF_meteo_DT_trim, 'P', Pa.PRJ, Xmin, Ymin, Xmax, Ymax)

        with TqdmCallback(desc=' -- Loading HD data for SFR-relevant layers and specified time range ...'):
            if load_HD:
                A_HD = A_HD_.sel(layer=l_SFR_Ls).sel(time=slice(start_date, end_date)).compute()
            if load_HD_RIV:
                A_HD_RIV = A_HD_RIV_.sel(layer=l_SFR_Ls).sel(time=slice(start_date, end_date)).compute()
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
                    L, R, C = reach_to_cell_id(reach, GDF_SFR)
                else:
                    parts = re.split(r'[,\s]+', In2.strip())  # Split by commas and/or whitespace
                    L, R, C = [int(j) for j in parts]
                    reach = GDF_SFR.loc[(GDF_SFR.k == L) & (GDF_SFR.i == R) & (GDF_SFR.j == C), 'reach'].values[0]
                X, Y = reach_to_XY(reach, GDF_SFR)

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
                    HD = pd.DataFrame({'time': HD_ts.time.values, 'head': HD_ts.values})

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
                    HD_RIV = pd.DataFrame({'time': HD_ts_RIV.time.values, 'head': HD_ts_RIV.values})

                    # required keys: plot_type, y. Rest are kwargs for the plot type (e.g., line dict for scatter, marker dict for bar, etc.)

                X_axis = DF_trim['time']

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
                            'y': HD['head'],
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
                            'y': HD_RIV['head'],
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
                            'y': DF_trim[f'{SFR_Out_Col_prefix}L{L}_R{R}_C{C}'],
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
                            'y': [GDF_SFR.loc[GDF_SFR['rno'] == reach, 'rtp'].values[0]] * len(DF_trim),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#ff0000', width=2, dash='dash'),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"RIV stage":<17}': {
                            'plot_type': go.Scatter,
                            'y': [round(float(get_value(A_RIV_Stg, X, Y, dx, dy)), 3)] * len(DF_trim),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#0000ff', width=3),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"RIV bottom":<17}': {
                            'plot_type': go.Scatter,
                            'y': [round(float(get_value(A_RIV_Btm, X, Y, dx, dy)), 3)] * len(DF_trim),
                            'kwargs': {
                                'mode': 'lines',
                                'line': dict(color='#0000ff', width=2, dash='dash'),
                                'hovertemplate': '%{y:3.3f} mNAP',
                            },
                        }
                    }
                )

                d_plot.update(
                    {
                        f'{"DRN elevation":<17}': {
                            'plot_type': go.Scatter,
                            'y': [round(float(get_value(A_DRN_Elv, X, Y, dx, dy)), 3)] * len(DF_trim),
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
                            'y': [round(float(get_value(A_TOP, X, Y, dx, dy, L=L)), 3)] * len(DF_trim),
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
                            'y': [round(float(get_value(A_BOT, X, Y, dx, dy, L=L)), 3)] * len(DF_trim),
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

                Pa_Out = Pa.PoP_Out_MdlN / f'SFR/stage_TS-reach{reach}.html'
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

                print('\n'.join(err_lines))
                print('Please try again.')
    sprint('🟢🟢🟢 - Finished SFR stage TS plotting.')
