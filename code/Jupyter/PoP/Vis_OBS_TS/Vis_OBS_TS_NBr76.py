"""
<span style="font-size:24px; font-family:'Roboto'; font-weight:bold;">
Script to PoP OBS TS file, to create visualizations.
</span><br>
To be used after the model Sim has finished, and the obs.csv file has been created.<br>
It was discovered that there are some mismatches between OBS and model cells (in NBr5). In NBr8, this was corrected. This script is designed to correct this. There are multiple cells connected to some wells now. The weighted average: h = ( T1*h1 + T2*h2 + ... ) / (T1 + T2 + ...), will be the main line on the plot.
"""

# %% 0. Imports
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import WS_Mdl.core.df  # noqa: F401
from plotly.subplots import make_subplots
from WS_Mdl.core import Mdl_N
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.metrics import Vld_Mtc
from WS_Mdl.imod.prj import r_with_OBS

# %% 1. Options
M = Mdl_N('NBr76')
PRJ, OBS = r_with_OBS(M.Pa.PRJ)
Pa_OBS_IPF = (M.Pa.PRJ.parent / OBS[-1].split(',')[-1].strip().strip("'")).resolve()
Pa_CSV = Path(str(M.Pa.MF6).replace('E:', 'G:')) / 'HD_OBS_NBr73_TS.csv'

# %% 2. Read and prepare OBS IPF file
DF_OBS = imod.formats.ipf.read(Pa_OBS_IPF)  # Read IPF file containing OBS HDs
DF_OBS = DF_OBS.ws.XY_to_RC(M, x='X', y='Y')


# %% 3. Read modelled HDs
DF_Mdl = pd.read_csv(Pa_CSV, index_col='time')  # Read CSV file containing modelled HDs
DF_Mdl.index = pd.to_datetime(M.SP_1st) + pd.to_timedelta(DF_Mdl.index, unit='D')


# %% 4. Prep Metrics
metrics = [
    Vld_Mtc('NSE', '-'),
    Vld_Mtc('RMSE', 'm'),
    Vld_Mtc('MAE', 'm'),
    Vld_Mtc('Correlation', '-'),
    Vld_Mtc('Bias Ratio', '-'),
    Vld_Mtc('Variability Ratio', '-'),
    Vld_Mtc('KGE', '-'),
]

DF_Mtc = pd.DataFrame(
    {
        'Metric': [metric.name for metric in metrics],
        'Value': np.nan,
        'Unit': [metric.unit for metric in metrics],
        'Formula': [metric.formula for metric in metrics],
    }
)

DF_Mtc_I = pd.DataFrame(columns=[i for i in DF_Mtc['Metric']]).astype(float)  # Empty for now
DF_Mtc_I.index.name = 'Obs_id'
DF_Mtc_I


# %% 5.0 Def HtML plot function + prep folder
def Plot1(MdlN, DF, id, adj_min, adj_max, DF_Pct, DF_Mtc_I, Pa_Fo_HTML, X, Y, L, R, C_1):
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.78, 0.22],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
        subplot_titles=['Time-Series Plot', 'Parity Plot', 'Percentile Plot'],
        specs=[[{'rowspan': 2}, {}], [None, {}]],
    )
    fig.add_trace(
        go.Scatter(
            x=DF['datetime'], y=DF['head'], mode='markers', name='Observed', marker=dict(size=3, color='#74c476')
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=DF['datetime'], y=DF[id], mode='lines', name='Simulated', line=dict(color='#005a1b')), row=1, col=1
    )  # Compute percentiles for head and simulated values
    fig.add_trace(
        go.Scatter(
            x=DF['head'],
            y=DF[id],
            mode='markers',
            marker=dict(size=4, color='#3a8448'),
            name='Scatter',
            hovertemplate='Observed: %{x}<br>Simulated: %{y}<extra></extra>',
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[adj_min, adj_max],
            y=[adj_min, adj_max],
            mode='lines',
            name='1:1 Line',
            line=dict(color='darkgrey', dash='dash'),
        ),
        row=1,
        col=2,
    )  # Make figure
    fig.add_trace(
        go.Scatter(x=DF_Pct['Percentile'], y=DF_Pct['Obs'], mode='lines', name='Observed', line=dict(color='#74c476')),
        row=2,
        col=2,
    )  # Less horizontal space
    fig.add_trace(
        go.Scatter(x=DF_Pct['Percentile'], y=DF_Pct['Sim'], mode='lines', name='Simulated', line=dict(color='#005a1b')),
        row=2,
        col=2,
    )
    stats_text = "<span style='font-family:Courier New; white-space:pre;'>NSE: {:>5.2f}  <br>RMSE:{:>5.2f} m<br>MAE: {:>5.2f} m<br>Cor: {:>5.2f}  <br>BR:{:>5.2f}  <br>VR: {:>5.2f}  <br>KGE: {:>5.2f}  </span>".format(
        DF_Mtc_I.loc[id, 'NSE'],
        DF_Mtc_I.loc[id, 'RMSE'],
        DF_Mtc_I.loc[id, 'MAE'],
        DF_Mtc_I.loc[id, 'Correlation'],
        DF_Mtc_I.loc[id, 'Bias Ratio'],
        DF_Mtc_I.loc[id, 'Variability Ratio'],
        DF_Mtc_I.loc[id, 'KGE'],
    )  # Create Plots for each Graph (TS, Parity, Pct)
    fig.add_annotation(
        text=stats_text,
        xref='x domain',
        yref='y domain',
        x=1,
        y=0,
        showarrow=False,
        font=dict(size=12, family='Courier New'),
        bgcolor='white',
        borderwidth=1,
        borderpad=5,
        align='left',
        row=1,
        col=2,
    )  # TS plot - Obs dots
    fig.update_yaxes(title_text='Head (mNAP)', tickformat='.2f', row=1, col=1)  # TS plot - Sim line
    fig.update_yaxes(title_text='Head (mNAP)', tickformat='.2f', row=2, col=2)
    tick_step = round((adj_max - adj_min) / 10, 1)  # Parity - dots
    tick_values = np.arange(adj_min, adj_max + tick_step, tick_step)  # Parity - 1:1 line
    tick_values = np.round(tick_values, 1)
    fig.update_xaxes(
        title_text='Observed Head (mNAP)',
        tickformat='.1f',
        row=1,
        col=2,
        range=[adj_min, adj_max],
        tickvals=tick_values,
    )  # Pct - Obs
    fig.update_yaxes(
        title_text='Simulated Head (mNAP)',
        tickformat='.1f',
        row=1,
        col=2,
        range=[adj_min, adj_max],
        tickvals=tick_values,
    )  # Pct - Sim
    fig.update_xaxes(title_text='Percentile (%)', tickformat='.1f', row=2, col=2)
    fig.add_annotation(
        text="<b>Simulated</b>  <span style='color:#005a1b;'>▬▬▬</span><br><b>Observed</b>        <span style='color:#74c476;'>●</span>",
        xref='x domain',
        yref='y domain',
        x=0,
        y=1,
        showarrow=False,
        font=dict(size=14),
        bgcolor='white',
        borderwidth=1,
        borderpad=5,
        align='left',
        row=1,
        col=1,
    )
    fig.add_annotation(
        text="<b>Deviation</b> <span style='color:#3a8448;'>      ●</span> <br><b>1:1</b>              <span style='color:darkgrey;'>━ ━ ━</span>",
        xref='x domain',
        yref='y domain',
        x=0,
        y=1,
        showarrow=False,
        font=dict(size=12),
        bgcolor='white',
        borderwidth=1,
        borderpad=5,
        align='left',
        row=1,
        col=2,
    )
    fig.add_annotation(
        text="<b>Simulated</b>  <span style='color:#005a1b;'>▬▬▬</span><br><b>Observed</b>        <span style='color:#74c476;'>●</span>",
        xref='x domain',
        yref='y domain',
        x=0,
        y=1,
        showarrow=False,
        font=dict(size=12),
        bgcolor='white',
        borderwidth=1,
        borderpad=5,
        align='left',
        row=2,
        col=2,
    )
    fig.update_layout(
        title=dict(
            text=f'<b>Groundwater Head Validation - {MdlN}</b><br><span style="font-size:14px; font-weight:normal;">id: {id} | X: {X}, Y: {Y}, L: {L}, R: {R}, C: {C_1}</span>',
            font=dict(size=20),
            y=0.98,
            x=0.5,
            xanchor='center',
        ),
        margin=dict(t=80, b=40, l=40, r=40),
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        showlegend=False,
    )
    fig.update_layout(legend=dict(font=dict(size=10)))
    fig.update_layout(
        hovermode='x unified',
        spikedistance=1000,
        xaxis_showspikes=True,
        yaxis_showspikes=False,
        xaxis_spikemode='across',
    )
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        margin=dict(autoexpand=True),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
    )
    print(f'Saving {id} ... ', end='')
    fig.write_html(Pa_Fo_HTML / f'{id}.HTML')
    print('completed!')


Pa_Fo_HTML_1 = M.Pa.PoP_Out_MdlN / 'GW_HD_OBS'
Pa_Fo_HTML_2 = M.Pa.PoP_Out_MdlN / 'GW_HD_OBS/problematic'
Pa_Fo_HTML_1.mkdir(parents=True, exist_ok=True)  # Make folder to store HTML files if it doesn't already exist.
Pa_Fo_HTML_2.mkdir(parents=True, exist_ok=True)  # Make folder to store HTML files if it doesn't already exist.


# %% 5.1 Make HTML Calibration plots
# n = 1

for ID in DF_Mdl.columns:  # [n : n + 1]:
    DF = DF_OBS.loc[(DF_OBS['Id'] == ID)].merge(
        right=DF_Mdl[ID], how='outer', left_on='datetime', right_index=True
    )  # Merge OBS and Mdl
    DF.index = pd.to_datetime(DF['datetime'])  # Set datetime as the index now that it's unique (per ID)
    DF_notNA = DF.loc[
        DF['head'].notna() & DF[ID].notna()
    ]  # Subset to rows where both OBS and Mdl have values, for percentile calculations

    # Extract info
    X, Y, L, R, C_1 = DF_OBS.loc[(DF_OBS['Id'] == ID)].iloc[0][['X', 'Y', 'L', 'R', 'C']]
    min_val, max_val = (
        np.floor(min(DF['head'].min(), DF[ID].min()) * 10) / 10,
        np.ceil(max(DF['head'].max(), DF[ID].max()) * 10) / 10,
    )
    buffer = (max_val - min_val) * 0.05
    adj_min, adj_max = (min_val - buffer, max_val + buffer)
    obs = DF_notNA['head'] if not DF_notNA['head'].empty else np.nan
    sim = DF_notNA[ID] if not DF_notNA[ID].empty else np.nan

    Pa_Fo_HTML_ = (
        Pa_Fo_HTML_2 if (np.isnan(obs).all()) or (np.isnan(sim).all()) else Pa_Fo_HTML_1
    )  # Store elsewhere if missing data

    # DT_common_min = max(DF.index.min(), DF[ID].index.min())
    # DT_common_max = min(DF.index.max(), DF[ID].index.max())
    for m in metrics:  # Calculate validation metrics and append to DF_Vld_Glb
        DF_Mtc_I.loc[ID, m.name] = m.compute(obs, sim) if ~np.isnan(obs).all() and ~np.isnan(sim).all() else np.nan

    Pctls = np.linspace(0, 100, 101)
    DF_Pct = pd.DataFrame({'Percentile': Pctls, 'Obs': np.percentile(obs, Pctls), 'Sim': np.percentile(sim, Pctls)})

    Plot1(
        M.MdlN, DF, ID, adj_min, adj_max, DF_Pct, DF_Mtc_I, Pa_Fo_HTML_, X, Y, L, R, C_1
    )  # Create and save HTML plot for the current OBS location)

    if (np.isnan(obs).all()) or (np.isnan(sim).all()):
        print(f'  X {ID} missing data or no overlap, stored in "problematic" folder.')

print('\n----- Finished creating HTML plots! -----')

# %% 6. Create GPKG
DF_GPKG = (
    DF_Mtc_I.round(3)
    .merge(right=DF_OBS[['Id', 'X', 'Y', 'L', 'R', 'C', 'path']], how='left', left_index=True, right_on='Id')
    .drop_duplicates(subset='Id')
    .dropna(subset=['NSE'])
)
DF_GPKG['path'] = DF_GPKG['Id'].apply(lambda x: f'file:///{(M.Pa.PoP_Out_MdlN / "GW_HD_OBS" / f"{x}.HTML").as_posix()}')
GDF_GPKG = gpd.GeoDataFrame(DF_GPKG, geometry=gpd.points_from_xy(DF_GPKG['X'], DF_GPKG['Y']), crs=CRS)
GDF_GPKG.to_file(M.Pa.PoP_Out_MdlN / f'GW_HD_OBS/GW_HD_OBS_Pnts_{M.MdlN}.gpkg')
