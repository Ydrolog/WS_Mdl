# %% Title
"""
Function to create Outlet flow TS plot. The total flow is comprised of the following components:
1. qrun: From MSW. | Always present.
2. DRN: OBS are added to each Sim before it starts. Those need to be aggregated for the catchment. | In SFR Sims, those are connected to the closest SFR cell via MVR.
3. RIV: OBS are added to each Sim before it starts. Those need to be aggregated for the catchment. | In SFR Sims (where a RIV isn't replaced completely), those are connected to the closest SFR cell via MVR.
4. SFR: Read for the outlet reach from the SFR OBS Out file.
"""

# %% Imports
from datetime import datetime as DT
from pathlib import Path
from time import time

import geopandas as gpd
import imod
import pandas as pd
import plotly.graph_objects as go

from WS_Mdl.core.defaults import CRS, Pa_WS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import sprint
from WS_Mdl.imod.msw.mete_grid import to_DF
from WS_Mdl.imod.msw.meteo import to_XA
from WS_Mdl.imod.pop.text import Agg_OBS
from WS_Mdl.imod.prj import r_with_OBS

# %% Options
Pa_clip = (
    Pa_WS / r'models\NBr\PoP\common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp'
)  # Area to clip MSW qrun to; this should be the catchment boundary of the outlet we're analyzing.

l_Sims = ['NBr100', 'NBr101', 'NBr102']

X_outlet, Y_outlet = 114213.79, 394950.96  # To get P for correct cell.

Pa_OBS_TS_xlsx = r'g:\models\NBr\other\BrabantseDelta\TS\Totaaldebiet-54689042-WNS2367_2025-07-31_11-18-44_02-00.xlsx'  # s/s where OBS TS is stored.

Pa_Out = Pa_WS / r'models\NBr\PoP\TS\Chaamse Beek Outlet Validation'  # Output folder for CSVs and plot.


# %% Define function
def stack_Out_TS(
    Pa_OBS_TS_xlsx: Path,
    Pa_clip_Shp: Path,
    l_Sims: list,
    X_outlet: float,
    Y_outlet: float,
):
    # %% Load OBS TS into DF
    DF_OBS = pd.read_excel(
        Pa_OBS_TS_xlsx,
        skiprows=12,
        names=['date', 'flow (m3/s)'],
    )
    DF_OBS['Observed'] = DF_OBS['flow (m3/s)'] * 86400
    DF_OBS.drop(columns=['flow (m3/s)'], inplace=True)
    DF_OBS['date'] = DF_OBS['date'].dt.normalize()
    # DF_OBS.describe()

    # %% Initialize empty DataFrames for each component
    d_DF = {}
    for i in ['DRN', 'RIV', 'SFR', 'qrun']:
        try:
            d_DF[i] = pd.read_csv(Pa_Out / f'Chaamse_Beek_Outlet-{i}_flows.csv', parse_dates=['date'])
            d_DF[i]['date'] = pd.to_datetime(d_DF[i]['date'], format='%d/%m/%Y', errors='coerce')
            sprint(f'🟢 - Initialized {i} DataFrame with OBS dates')
        except Exception:
            d_DF[i] = pd.DataFrame()

    # %% Read and prep catchment boundary
    GDF_CB = gpd.read_file(Pa_clip_Shp)

    if (GDF_CB.crs != CRS) or (GDF_CB.crs is None):  # Set default CRS
        GDF_CB = GDF_CB.set_crs(CRS, allow_override=True)

    # %% Iterate over Sims and append
    start_time = time()
    for MdlN in l_Sims[:]:
        print(MdlN, f'start time: {round(time() - start_time, 2)} s')
        M = Mdl_N(MdlN)

        # DRN + RIV
        for Pkg in ['DRN', 'RIV']:
            if MdlN not in d_DF[Pkg].columns:  # If this Sim's OBS haven't been appended yet
                try:
                    Pa_OBS_Agg = (M.Pa.MdlN if M.V == 'imod5' else M.Pa.MF6) / f'OBS_Agg/{Pkg}_OBS_Agg_{MdlN}.csv'
                    if Pa_OBS_Agg.exists():
                        DF = pd.read_csv(Pa_OBS_Agg, parse_dates=['date'])[['date', 'SUM']]
                        method = 'Load'
                    else:  # If Agg file doesn't exist, create it from Out.
                        DF = Agg_OBS(MdlN, Pkg, True, True)[['date', 'SUM']]
                        method = 'Agg_OBS'

                    DF['date'] = pd.to_datetime(DF['date'], format='%d/%m/%Y', errors='coerce')
                    DF.rename(columns={'SUM': MdlN}, inplace=True)
                    DF[MdlN] = DF[MdlN] * (-1)  # Correct sign
                    if d_DF[Pkg].empty:
                        d_DF[Pkg] = DF
                    else:
                        d_DF[Pkg] = d_DF[Pkg].merge(DF, on='date', how='outer')
                    sprint(
                        f'🟢 - Sucessfully appended {MdlN} {Pkg}. Method: {method} + Merge.',
                        indent=1,
                    )
                except Exception as e:
                    DF = pd.DataFrame({'date': [], 'SUM': []})
                    DF['date'] = pd.to_datetime(DF['date'], format='%d/%m/%Y', errors='coerce')
                    sprint(f'  🔴 - Failed to read {MdlN} {Pkg}. Error:\n{e}')

        # SFR
        if MdlN not in d_DF['SFR'].columns:
            try:
                DF = pd.read_csv(
                    M.Pa.Sim_In / f'{MdlN}.SFR6.obs.output.csv',
                    usecols=['time', 'OUTLET_DOWNSTREAM-FLOW'],
                )
                DF['date'] = DT.strptime(str(M.INI.SDATE), '%Y%m%d') + pd.to_timedelta(DF['time'] - 1, unit='D')
                DF.rename(columns={'OUTLET_DOWNSTREAM-FLOW': MdlN}, inplace=True)
                DF[MdlN] = DF[MdlN] * (-1)
                DF = DF[['date', MdlN]]
                if d_DF['SFR'].empty:
                    d_DF['SFR'] = DF
                else:
                    d_DF['SFR'] = d_DF['SFR'].merge(DF, on='date', how='outer')
                sprint(f'  🟢 - Sucessfully appended {MdlN} SFR')
            except Exception as e:
                sprint(f'  🔴 - Failed to read {MdlN} SFR. Error:\n{e}')
        else:
            sprint(f'  ⚪️ - {MdlN} SFR already exists, skipping.')

        # MSW qrun
        if MdlN not in d_DF['qrun'].columns:
            try:
                # Load qrun IDFs to xarray
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
                DF = pd.DataFrame(
                    {
                        'date': DA_qrun_clip_sum['time'].values,
                        f'{MdlN}': DA_qrun_clip_sum.values,
                    }
                )  # Convert from m3/s to m3/d
                if d_DF['qrun'].empty:
                    d_DF['qrun'] = DF
                else:
                    d_DF['qrun'] = d_DF['qrun'].merge(DF, on='date', how='outer')

                sprint(f'  🟢 - Sucessfully appended {MdlN} MSW qrun')
            except Exception as e:
                sprint(f'  🔴 - Failed to read {MdlN} MSW qrun. Error:\n{e}')
        else:
            sprint(f'  ⚪️ - {MdlN} MSW qrun already exists, skipping.')

    # %% Save individual component CSVs
    for i in ['DRN', 'RIV', 'SFR', 'qrun']:
        print(d_DF[i].columns)
        d_DF[i].set_index('date', inplace=True)
        print(Pa_Out / f'Chaamse_Beek_Outlet-{i}_flows.csv')
        d_DF[i].to_csv(Pa_Out / f'Chaamse_Beek_Outlet-{i}_flows.csv')

    # %% SUM for MdlN
    for MdlN in l_Sims[:]:
        for k, v in d_DF.items():
            if MdlN not in v.columns:
                sprint(f'  ⚪️ - {MdlN} missing from {k}, filling with NaNs.')
                v[MdlN] = pd.NA
            d_DF[k] = d_DF[k].apply(pd.to_numeric, errors='coerce')

    DF_Agg = (
        d_DF['DRN']
        .fillna(0)
        .add(d_DF['RIV'].fillna(0), fill_value=0)
        .add(d_DF['SFR'].fillna(0), fill_value=0)
        .add(d_DF['qrun'].fillna(0), fill_value=0)
    )
    date_min, date_max = DF_Agg.index.min(), DT.strptime('2001-12-31', '%Y-%m-%d')
    print(date_min, date_max)
    print(type(date_min), type(date_max))

    # %% Load Precipitation
    PRJ, OBS = r_with_OBS(M.Pa.PRJ)
    DF_meteo = to_DF(PRJ)
    DF_meteo = DF_meteo.loc[
        (DF_meteo['DT'] <= date_max) & (DF_meteo['DT'] >= date_min)
    ]  # Clip data range to speed up loading
    # %%
    A_P = to_XA(DF_meteo, 'P', MdlN, clip=False)

    # %% Select outlet P and convert to DF
    DA_P = A_P.sel(x=X_outlet, y=Y_outlet, method='nearest')
    Vals_P = DA_P.data.compute() if hasattr(DA_P.data, 'compute') else DA_P.data

    DF_P = pd.DataFrame(
        {
            'date': pd.to_datetime(DA_P['time'].values),
            'P (mm/d)': Vals_P,
        }
    )

    # %% Aggregate
    DF_SUM = DF_P.merge(DF_OBS, how='outer', on='date').merge(DF_Agg, on='date', how='outer')

    # %% Rename columns
    DF_SUM.rename(columns={'P (mm/d)': 'P'}, inplace=True)
    DF_SUM = DF_SUM[['date', 'P', 'Observed'] + l_Sims]
    DF_SUM.to_csv(Pa_Out / 'Chaamse_Beek_Outlet_SUM_flows.csv', index=False)

    # %%  Plot
    DF_plot = DF_SUM.loc[(DF_SUM['date'] >= date_min) & (DF_SUM['date'] <= date_max)].copy()

    # Prepare plotting frame: use `date` as x and all other columns as y-series.
    DF_plot['date'] = pd.to_datetime(DF_plot['date'], errors='coerce')

    Y_Cols = [C for C in DF_plot.columns if C != 'date']
    if not Y_Cols:
        raise ValueError("DF_OBS must contain at least one timeseries column besides 'date'.")

    # Coerce all series to numeric.
    for C in Y_Cols:
        DF_plot[C] = pd.to_numeric(DF_plot[C], errors='coerce')

    # Keep only plottable series.
    Y_Cols = [C for C in Y_Cols if DF_plot[C].notna().any()]
    if not Y_Cols:
        raise ValueError('No numeric timeseries columns available in DF_OBS for plotting.')

    # Split series: precipitation on top panel, flow series on bottom panel.
    P_Cols = [
        C
        for C in Y_Cols
        if str(C).strip().lower() in {'p', 'p (mm/d)', 'precipitation', 'precipitation (mm/d)'}
        or 'precip' in str(C).lower()
    ]
    Flow_Cols = [C for C in Y_Cols if C not in P_Cols]

    fig = go.Figure()

    # Auto-scaling for bottom panel, similar to SFR_reach_TS.
    Y_Vals = []
    for C in Flow_Cols:
        S = DF_plot[C]
        Y_Vals.extend(S.dropna().values.tolist())

    if Y_Vals:
        Y_Min, Y_Max = min(Y_Vals), max(Y_Vals)
        Y_Pad = (Y_Max - Y_Min) * 0.05 if (Y_Max - Y_Min) > 0 else 0.5
        Y_Range = [Y_Min - Y_Pad, Y_Max + Y_Pad]
    else:
        Y_Range = [0, 1]

    for C in Flow_Cols:
        fig.add_trace(
            go.Scatter(
                x=DF_plot['date'],
                y=DF_plot[C],
                mode='lines',
                name=str(C),
                line={'width': 2},
                yaxis='y',
            )
        )

    for C in P_Cols:
        fig.add_trace(
            go.Bar(
                x=DF_plot['date'],
                y=DF_plot[C],
                name=str(C),
                marker={
                    'color': 'rgb(65, 105, 225)',
                    'opacity': 1,
                    'line': {'width': 0},
                },
                yaxis='y2',
            )
        )

    fig.update_traces(opacity=1, selector={'type': 'bar'})

    fig.update_layout(
        title=dict(
            text=f"Chaamse Beek Outlet - Timeseries<br><span style='font-size: 14px; color: gray;'>{date_min} to {date_max}</span>",
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top',
        ),
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font=dict(family='Consolas, monospace', size=12),
        ),
        hovermode='x unified',
        hoverdistance=-1,
        spikedistance=-1,
        hoverlabel=dict(
            namelength=-1,
            bgcolor='white',
            bordercolor='gray',
            font=dict(family='Consolas, monospace'),
            align='right',
        ),
        template='plotly_white',
        margin=dict(t=80, l=60, r=40, b=40),
        xaxis=dict(
            domain=[0, 1],
            anchor='free',
            position=0,
            dtick='M1',
            tickformat='%b %Y',
            tickangle=-90,
            hoverformat='%d %b %Y',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            showgrid=True,
        ),
        yaxis=dict(
            title_text='Flow rate (m3/d)',
            domain=[0, 0.68],
            range=Y_Range,
            nticks=15,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            showgrid=True,
            fixedrange=False,
        ),
        yaxis2=dict(
            title_text='Precipitation (mm/d)',
            domain=[0.72, 1],
            rangemode='tozero',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            showgrid=True,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
            fixedrange=False,
        ),
        barmode='overlay',
    )

    Pa_Htm = Path(Pa_Out / f'Outlet_TS_{date_min.strftime("%Y%m%d")}-{date_max.strftime("%Y%m%d")}.html')
    fig.write_html(Pa_Htm, include_plotlyjs='cdn', full_html=True)
    fig.show()

    print(f'Saved Plotly HTML: {Pa_Htm.resolve()}')


if __name__ == '__main__':
    stack_Out_TS(Pa_OBS_TS_xlsx, Pa_clip, l_Sims, X_outlet, Y_outlet)
