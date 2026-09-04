# To compare LHM HDs with previous HDs, and convert to NBr Mdl layers (LHM has 8, NBr has 37)

# %% Imports
from pathlib import Path

import imod
import numpy as np
import pandas as pd
import plotly.express as px
import WS_Mdl.core.df  # Noqa: F401
import xarray as xra
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import sprint
from WS_Mdl.imod.prj import r_with_OBS
from WS_Mdl.xr.spatial import clip_Mdl_area

# %% Options
MdlN = 'NBr111'
MdlN_B = 'NBr101'

# %%
MB = Mdl_N(MdlN_B)
M = Mdl_N(MdlN)

PRJ, OBS = r_with_OBS(MB.Pa.PRJ)

l_Pa_CHD1 = [
    i for i in (MB.Pa.In / 'CHD/NBr1').glob('*.idf') if ('HEAD_2017' not in i.name) and ('HEAD_2018' not in i.name)
]
l_Pa_CHD2 = [i for i in (MB.Pa.In / 'CHD/NBr5').glob('*.idf') if ('HEAD_20130428' not in i.name)]


# %% Load old CHDs - 1
CHD1 = clip_Mdl_area(imod.idf.open(l_Pa_CHD1, pattern='{name}_{time}_L{layer}_NBr1'), MdlN_B, buffer=1000)

# %% Load old CHDs - 2
CHD2 = clip_Mdl_area(imod.idf.open(l_Pa_CHD2, pattern='{name}_{time}_L{layer}_NBr5'), MdlN_B, buffer=1000)

# %% Load old CHDs - Merge
CHD1 = CHD1.interp_like(CHD2, method='nearest')
CHD = xra.concat([CHD1, CHD2], dim='time').sortby('time')

# %% Load OBS
Pa_OBS_IPF = (
    M.Pa.PRJ.parent / OBS[-1].split(',')[-1].strip().strip("'")
).resolve()  # Combines PRJ path with OBS relative path
DF_OBS = imod.formats.ipf.read(Pa_OBS_IPF)  # Read IPF file containing OBS HDs
DF_OBS = DF_OBS.ws.XY_to_RC(MB, x='X', y='Y')

# %% Load LHM HDs, TOP & BOT
HD = clip_Mdl_area(
    imod.idf.open(M.Pa.In / 'CHD/LHM/heads/head_*_l*.idf', pattern='{name}_{time}_l{layer}'), MdlN_B, buffer=1000
)
TOP = clip_Mdl_area(imod.idf.open(M.Pa.In / 'CHD/LHM/top/TOP_L*.idf', pattern='{name}_L{layer}'), MdlN_B, buffer=1000)
BOT = clip_Mdl_area(imod.idf.open(M.Pa.In / 'CHD/LHM/bot/BOT_L*.idf', pattern='{name}_L{layer}'), MdlN_B, buffer=1000)

# %%
GRB = imod.mf6.read_grb(MB.Pa.GRB)

# Analysis
# %% CHD/ GRB
for k, v in GRB.items():
    print(k)

    if isinstance(v, str) or isinstance(k, int):
        print(v)
    else:
        try:
            print(v.shape)
        except:
            print()
    print('-----')

# %%
CHD.x.values, CHD.y.values

# %% LHM
HD.x.values, HD.y.values


# Map each CHD layer to the LHM layer with the nearest vertical midpoint.

# %%
GRB_mid = (
    GRB['bottom']
    + xra.concat(
        [GRB['top'].expand_dims(layer=[CHD.layer.values[0]]), GRB['bottom'].sel(layer=CHD.layer[:-1])],
        dim='layer',
    ).assign_coords(layer=CHD.layer)
) / 2

# %% Calc LHM midpoints
MID = (TOP + BOT) / 2
HD_src = HD.interp(x=GRB_mid.x, y=GRB_mid.y, method='nearest')
MID_src = MID.interp(x=GRB_mid.x, y=GRB_mid.y, method='nearest')

# %% Iterate
l_HD_L = []
for L in CHD.layer.values:
    Dist = abs(MID_src - GRB_mid.sel(layer=L))
    Has_source = Dist.notnull().any('layer')
    Src_i = Dist.fillna(float('inf')).argmin('layer')

    HD_L = sum(
        HD_src.sel(layer=Src_L).where(Src_i == Src_i_Val, 0) for Src_i_Val, Src_L in enumerate(HD.layer.values)
    ).where(Has_source)
    l_HD_L.append(HD_L.expand_dims(layer=[L]))

HD_ = xra.concat(l_HD_L, dim='layer')
HD_ = HD_.where(GRB['idomain'].sel(layer=CHD.layer) > 0)

# %% Trim time
CHD = CHD.sel(time=slice(DF_OBS.datetime.min(), DF_OBS.datetime.max()))
HD_ = HD_.sel(time=slice(DF_OBS.datetime.min(), DF_OBS.datetime.max()))

# %% Prepare one location per observation ID
DF_Points = DF_OBS.drop_duplicates('Id').set_index('Id')

Ids = DF_Points.index.to_numpy()

Point = {'point': Ids}

X_Indexer = xra.DataArray(
    DF_Points.X.to_numpy(),
    dims='point',
    coords=Point,
)

Y_Indexer = xra.DataArray(
    DF_Points.Y.to_numpy(),
    dims='point',
    coords=Point,
)

L_Indexer = xra.DataArray(
    DF_Points.L.to_numpy(),
    dims='point',
    coords=Point,
)

# Vectorized selection: result dimensions are (time, point)
CHD_Points = CHD.sel(
    x=X_Indexer,
    y=Y_Indexer,
    layer=L_Indexer,
    method='nearest',
)

HD_Points = HD_.sel(
    x=X_Indexer,
    y=Y_Indexer,
    layer=L_Indexer,
    method='nearest',
)

# Perform all expensive work once, before plotting
sprint('--- Loading all observation time series', set_time=True)
CHD_Points.load()
HD_Points.load()
sprint('🟢', print_time=True)

# %% Save plots for each OBS


def model_at_observation_times(Model_HD, Obs_Times):
    """Interpolate a model series without extrapolating beyond its valid period."""
    Result = np.full(len(Obs_Times), np.nan)
    Model = pd.DataFrame(
        {
            'datetime': pd.to_datetime(Model_HD.time.values),
            'head': np.asarray(Model_HD.values).squeeze(),
        }
    ).dropna()
    Model = Model.groupby('datetime', as_index=False)['head'].mean().sort_values('datetime')

    if len(Model) < 2:
        return Result

    Available = Obs_Times.between(Model.datetime.iloc[0], Model.datetime.iloc[-1]).to_numpy()
    Origin = Model.datetime.iloc[0]
    Model_Days = (Model.datetime - Origin).dt.total_seconds().to_numpy() / 86400
    Obs_Days = (Obs_Times[Available] - Origin).dt.total_seconds().to_numpy() / 86400
    Result[Available] = np.interp(Obs_Days, Model_Days, Model['head'].to_numpy())
    return Result


def paired_rmse_at_observation_times(CHD_HD, LHM_HD, DF):
    """Calculate both RMSEs using the same observation dates."""
    Obs = DF[['datetime', 'head']].copy()
    Obs['datetime'] = pd.to_datetime(Obs['datetime'], errors='coerce')
    Obs['head'] = pd.to_numeric(Obs['head'], errors='coerce')
    Obs = Obs.dropna().sort_values('datetime')

    if Obs.empty:
        return np.nan, np.nan, np.nan, np.nan, 0

    CHD_At_Obs = model_at_observation_times(CHD_HD, Obs.datetime)
    LHM_At_Obs = model_at_observation_times(LHM_HD, Obs.datetime)
    Shared = np.isfinite(CHD_At_Obs) & np.isfinite(LHM_At_Obs)
    if not Shared.any():
        return np.nan, np.nan, np.nan, np.nan, 0

    Observed = Obs['head'].to_numpy()[Shared]
    Error_CHD = CHD_At_Obs[Shared] - Observed
    Error_LHM = LHM_At_Obs[Shared] - Observed
    RMSE_CHD = np.sqrt(np.mean(Error_CHD**2))
    RMSE_LHM = np.sqrt(np.mean(Error_LHM**2))
    ME_CHD = np.mean(Error_CHD)
    ME_LHM = np.mean(Error_LHM)
    return RMSE_CHD, RMSE_LHM, ME_CHD, ME_LHM, int(Shared.sum())


RMSE_Rows = []

for Id in Ids:
    sprint(f'--- Plotting Id: {Id}', set_time2=True)

    DF = DF_OBS.loc[DF_OBS.Id == Id].sort_values('datetime')
    Point_Info = DF_Points.loc[Id]
    X, Y, L = Point_Info.X, Point_Info.Y, Point_Info.L

    # These are now selections from arrays already held in memory
    CHD_ = CHD_Points.sel(point=Id)
    HD__ = HD_Points.sel(point=Id)

    RMSE_CHD, RMSE_LHM, ME_CHD, ME_LHM, N = paired_rmse_at_observation_times(CHD_, HD__, DF)
    RMSE_Rows.append(
        {
            'Id': Id,
            'L': L,
            'X': X,
            'Y': Y,
            'RMSE_CHD': RMSE_CHD,
            'RMSE_LHM': RMSE_LHM,
            'ME_CHD': ME_CHD,
            'ME_LHM': ME_LHM,
            'N': N,
        }
    )

    fig = px.line(DF, x='datetime', y='head')
    fig.data[0].update(
        name='OBS',
        showlegend=True,
        mode='lines+markers',
        line=dict(color='#238b45', width=2),
        marker=dict(size=2),
        hovertemplate='%{y:.2f} m<extra></extra>',
    )

    # Draw LHM first so the dashed CHD trace stays visible when they overlap.
    fig.add_scatter(
        x=HD__.time.values,
        y=HD__.values,
        mode='lines',
        name='LHM',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{y:.2f} m<extra></extra>',
    )

    fig.add_scatter(
        x=CHD_.time.values,
        y=CHD_.values,
        mode='lines',
        name='CHD',
        line=dict(color='#d62728', width=2),
        connectgaps=True,
        hovertemplate='%{y:.2f} m<extra></extra>',
    )

    fig.update_layout(
        title=dict(
            text=f'Id: {Id} | L: {L:g} | X: {X:,.0f} | Y: {Y:,.0f}',
            x=0.5,
            xanchor='center',
        ),
        hovermode='x unified',
        hoverdistance=-1,
        xaxis=dict(
            title='Date',
            hoverformat='%d-%b-%Y',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
        ),
        yaxis_title='Head (m)',
        legend_title_text='',
    )

    fig.write_html(
        f'plots/{Id}.html',
        include_plotlyjs='directory',
    )

# %%
DF_RMSE = pd.DataFrame(RMSE_Rows)

DF_RMSE['link'] = DF_RMSE.Id.apply(lambda x: f'=HYPERLINK("{Path("plots").resolve() / (x + ".html")}", "{x}")')
DF_RMSE.to_csv('plots/RMSE.csv', index=False)


# %%
