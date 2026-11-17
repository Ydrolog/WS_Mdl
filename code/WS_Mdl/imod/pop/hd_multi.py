
# %% Imports
import os
from concurrent.futures import ProcessPoolExecutor as PPE
from datetime import datetime as DT
from pathlib import Path

import imod
import numpy as np
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import Sep, green, sprint
from WS_Mdl.imod.idf import HD_Out_to_DF
from WS_Mdl.imod.prj import r_with_OBS
from WS_Mdl.xr.convert import to_TIF

__all__ = [
    'HD_IDF_Agg_to_TIF',
    # 'HD_Pctl_Diffs',
    'c_HD_Bin_AVGs',
    'c_HD_Bin_Daily',
    'c_HD_Bin_Yearly',
    'c_HD_Bin_Monthly',
    'c_HD_Bin_GXGs',
    'c_HD_Bin_Pctls',
    'c_HD_Pctls',
    'o_HD_Bin_Multi',
    'p_HD_OBS_TS',
    'L_PCT',
]


# %% Percentile levels (shared by every function that calculates percentiles)
def _make_l_Pct(lo: int = 25, hi: int = 75, step_mid: int = 5) -> tuple:
    """
    Percentile levels as fractions: every 1 % below `lo` and above `hi`, every `step_mid` % in between
    (lo and hi included). Built from whole percentages, so there is no floating-point drift.
    Default: P01-P24, P25, P30, ..., P70, P75, P76-P99 (59 levels).
    """
    pct = list(range(1, lo)) + list(range(lo, hi + 1, step_mid)) + list(range(hi + 1, 100))
    return tuple(p / 100 for p in pct)


L_PCT = _make_l_Pct()


def _P_label(q: float) -> str:
    """0.29 -> 'P29'. Uses round(), because int(0.29 * 100) gives 28 (0.29 * 100 = 28.999...)."""
    return f'P{round(q * 100):02d}'




# %%
def HD_IDF_Agg_to_TIF(
    MdlN: str,
    rules=None,
    N_cores: int | None = None,
    CRS: str = CRS,
    Gp: tuple[str, str] = ('year', 'month'),
    Agg_F: str = 'mean',
):
    """
    General wrapper to:
      1) read all IDF metadata into a DataFrame,
      2) filter by `rules`,
      3) add any needed Gp columns (season, Hy_year, quarter),
      4) group by `Gp`,
      5) for each group, apply `agg_func` along time and write a single‐band TIFF.

    Parameters
    ----------
    MdlN : str
        Model name (e.g. 'NBr13').
    rules : None or str
        A pandas-query string to subset/filter the IDF-DF before Gp (e.g. "(L == 1)").
    N_cores : int or None
        Number of worker processes for parallel execution. By default: None → use (cpu_count() - 2).
    CRS : str
        Coordinate reference system for the output TIFs. By default: G.CRS.
    Gp : tuple of str
        Which DataFrame columns to group by. Common examples:
        - ('year', 'month')        → monthly aggregates
        - ('season_year', 'season')→ seasonal aggregates
        - ('Hy_year',)             → hydrological-year aggregates
        - ('year', 'quarter')      → quarterly aggregates
    agg_func : str
        Name of the aggregation method to call on the xarray.DataArray (e.g. 'mean','min','max','median').
        This must exactly match a DataArray method (e.g. XA.mean(dim='time')).
    """

    def _HD_IDF_Agg_to_TIF_process(paths, Agg_F, Pa_Out, CRS, params):
        """
        Only for use within HD_IDF_Mo_Avg_to_MBTIF - to utilize multiprocessing.
        Reads IDFs, aggregates along time, writes each layer as a single-band TIF.
        """
        Pa_Out = Path(Pa_Out)  # ensure it's a Path object for consistent handling

        XA = imod.formats.idf.open(paths)
        XA_agg = getattr(XA, Agg_F)(dim='time')
        base = Pa_Out[:-4]  # strip “.tif”
        for layer in XA_agg.layer.values:
            DA = XA_agg.sel(layer=layer).drop_vars('layer')
            Out = f'{base}_L{layer}.tif'
            d_MtDt = {
                f'{Agg_F}': {
                    'AVG': float(DA.mean().values),
                    'coordinates': XA.coords,
                    'variable': Pa_Out.stem,  # name without suffix
                    'details': f'Calculated using WS_Mdl.geo.py using the following params: {params}',
                }
            }

            to_TIF(DA, Out, d_MtDt, CRS=CRS)
        return f'{base.name} 🟢 '

    sprint(Sep)
    sprint(f'*** {MdlN} *** - HD_IDF_Agg_to_TIF\n')

    # 1. Get paths
    M = Mdl_N(MdlN)
    Pa_PoP, Pa_HD = M.Pa.PoP, M.Pa.Out_HD

    # 2. Read the IDF files to DF. Add extracols. Apply rules. Group.
    DF = HD_Out_to_DF(Pa_HD)
    if rules:
        DF = DF.query(rules)
    DF_Gp = DF.groupby(Gp)['path']

    # 3. Prep Out Dir
    Pa_Out_Dir = Pa_PoP / f'Out/{MdlN}/HD_Agg'
    Pa_Out_Dir.mkdir(parents=True, exist_ok=True)

    # 4. Decide N of cores
    if N_cores is None:
        N_cores = max(
            os.cpu_count() - 2, 1
        )  # Leave 2 cores free for other tasks by default. If there aren't enough cores available, set to 1.

    # 5. Launch one job per group
    start = DT.now()
    with PPE(max_workers=N_cores) as E:
        futures = []
        for Gp_keys, paths in DF_Gp:
            group_name = HD_Agg_name(
                Gp_keys, Gp
            )  # user‐defined helper to turn keys → a nice string, e.g. "2010_1" or "2020_Winter"

            # we’ll write one single‐band GeoTiff per group
            Pa_Out = Pa_Out_Dir / f'HD_{group_name}_{MdlN}.tif'

            params = {
                'MdlN': str(MdlN),
                'N_cores': str(N_cores),
                'CRS': str(CRS),
                'rules': str(rules),
            }

            futures.append(
                E.submit(
                    _HD_IDF_Agg_to_TIF_process,
                    paths=list(paths),
                    Agg_F=Agg_F,
                    Pa_Out=Pa_Out,
                    CRS=CRS,
                    params=params,
                )
            )

        for f in futures:  # wait & report
            sprint('\t', f.result(), 'elapsed:', DT.now() - start)

    sprint(f'🟢🟢🟢 | Total elapsed time: {DT.now() - start}')
    sprint(Sep)


def HD_Agg_name(group_keys, grouping):
    if not isinstance(group_keys, (tuple, list)):
        group_keys = (group_keys,)

    if grouping == ['year', 'month']:  # year & month → "YYYYMM"
        year, month = group_keys
        return f'{year}{month:02d}'

    if grouping == ['month']:  # month alone → "MM"
        (month,) = group_keys
        return f'{month:02d}'

    if grouping == ['year']:  # year alone → "YYYY"
        (year,) = group_keys
        return str(year)

    if grouping == ['season_year', 'season']:  # season_year & season → "YYYY_Season"
        season_year, season = group_keys
        return f'{season_year}_{season}'

    if grouping == ['season']:  # season alone → "Season"
        (season,) = group_keys
        return season

    if grouping == ['water_year']:  # water_year → "WYYY"
        (wy,) = group_keys
        return f'WY{wy}'

    if grouping == ['year', 'quarter']:  # year & quarter → "YYYY_Q#"
        year, quarter = group_keys
        return f'{year}_{quarter}'

    if grouping == ['quarter']:  # quarter alone → "Q#"
        (quarter,) = group_keys
        return quarter

    return '_'.join(str(k) for k in group_keys)  # fallback: join all keys with underscore


def p_HD_OBS_TS(MdlN, MdlN_B=True, MdlN_Pa_MF6=None, MdlN_B_Pa_MF6=None):
    """
    Reads Mdl Out TS files (CSVs) for S and B (if B not False) and Obs data, then creates HTML plots in the PoP Out folder.
    """
    sprint(Sep)
    sprint('----- p_HD_OBS_TS initiated -----', style=green)

    # %% 1. Initial
    sprint('--- Loading data...')
    sprint(' -- Creating Mdl_N instances.', end='')
    import geopandas as gpd
    import pandas as pd
    import WS_Mdl.core.df  # noqa: F401
    from WS_Mdl.core.metrics import Vld_Mtc

    M = Mdl_N(MdlN)
    MB = Mdl_N(MdlN_B) if isinstance(MdlN_B, str) else Mdl_N(M.B) if MdlN_B is True else M
    if MdlN_Pa_MF6:
        M.Pa.MF6 = Path(MdlN_Pa_MF6)
    if MdlN_B_Pa_MF6:
        MB.Pa.MF6 = Path(MdlN_B_Pa_MF6)
    PRJ, OBS = r_with_OBS(M.Pa.PRJ)
    sprint('🟢')

    # %% 2. Read Obs
    sprint(' -- Loading OBS.', end='')
    Pa_OBS_IPF = (
        M.Pa.PRJ.parent / OBS[-1].split(',')[-1].strip().strip("'")
    ).resolve()  # Combines PRJ path with OBS relative path
    DF_OBS = imod.formats.ipf.read(Pa_OBS_IPF)  # Read IPF file containing OBS HDs
    DF_OBS = DF_OBS.ws.XY_to_RC(M, x='X', y='Y')

    # %% 3. Read modelled HDs
    DF_M = pd.read_csv(
        next(iter(M.Pa.MF6.rglob('HD_OBS_Pnt*.csv'))), index_col='time'
    )  # Read CSV file containing modelled HDs. Assumes 1 matching file
    DF_M.index = pd.to_datetime(M.SP_1st) + pd.to_timedelta(DF_M.index, unit='D')
    if MB:
        DF_MB = pd.read_csv(
            next(iter(MB.Pa.MF6.rglob('HD_OBS_Pnt*.csv'))), index_col='time'
        )  # Read CSV file containing modelled HDs for B. Assumes 1 matching file
        DF_MB.index = pd.to_datetime(MB.SP_1st) + pd.to_timedelta(DF_MB.index, unit='D')
    sprint('🟢')

    # %% 4. Def HtML plot function + prep folder
    sprint('--- Plotting ...')

    def Plot1(MdlN_S, MdlN_B, DF, Id, adj_min, adj_max, DF_Pct, DF_Mtc, Pa_Fo_HTML, X, Y, L, R, C_1):
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        col_s = '#74c476'  # Scenario = light green
        col_b = '#005a1b'  # Baseline = dark green
        col_obs = '#238b45'  # Observed = mid green
        col_11 = 'darkgrey'

        x_ts = pd.to_datetime(DF['datetime'] if 'datetime' in DF.columns else DF.index)
        x_ts = pd.Series(x_ts, index=DF.index)
        x_ts_fmt = x_ts.dt.strftime('%d-%b-%Y')

        def hover_line(label, value, date=None):
            if pd.isna(value):
                return None

            date_txt = f' | {date}' if date is not None else ''
            return f'{label + ":":<10} {value:6.2f} m{date_txt}'

        def hover_box(lines):
            lines = [line for line in lines if line is not None]
            return "<span style='font-family:Courier New; white-space:pre;'>" + '<br>'.join(lines) + '</span>'

        obs_mask = DF['head'].notna()
        DF_Obs_Nearest = pd.merge_asof(
            pd.DataFrame({'x': x_ts}).sort_values('x'),
            pd.DataFrame(
                {
                    'obs_x': x_ts[obs_mask],
                    'obs_val': DF.loc[obs_mask, 'head'],
                }
            ).sort_values('obs_x'),
            left_on='x',
            right_on='obs_x',
            direction='nearest',
        )

        ts_hover_text = [
            hover_box(
                [
                    hover_line(
                        'Observed',
                        obs_val,
                        obs_x.strftime('%d-%b-%Y') if pd.notna(obs_x) else None,
                    ),
                    hover_line(MdlN_S, s, date),
                    hover_line(MdlN_B, b, date),
                ]
            )
            for obs_val, obs_x, s, b, date in zip(
                DF_Obs_Nearest['obs_val'],
                DF_Obs_Nearest['obs_x'],
                DF[MdlN_S],
                DF[MdlN_B],
                x_ts_fmt,
            )
        ]

        parity_hover_text = [
            hover_box(
                [
                    hover_line('Observed', obs),
                    hover_line(MdlN_S, s),
                    hover_line(MdlN_B, b),
                ]
            )
            for obs, s, b in zip(DF['head'], DF[MdlN_S], DF[MdlN_B])
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.68, 0.32],
            row_heights=[0.5, 0.5],
            vertical_spacing=0.12,
            horizontal_spacing=0.05,
            subplot_titles=['Time-Series Plot', 'Parity Plot', 'Percentile Plot'],
            specs=[[{'rowspan': 2}, {}], [None, {}]],
        )

        # Time-series plot
        fig.add_trace(
            go.Scatter(
                x=x_ts,
                y=DF['head'],
                mode='markers',
                name='Observed',
                marker=dict(size=3, color=col_obs),
                hoverinfo='skip',
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_ts,
                y=DF[MdlN_B],
                mode='lines',
                name=MdlN_B,
                line=dict(color=col_b),
                connectgaps=False,
                hoverinfo='skip',
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_ts,
                y=DF[MdlN_S],
                mode='lines',
                name=MdlN_S,
                line=dict(color=col_s),
                connectgaps=False,
                hoverinfo='skip',
            ),
            row=1,
            col=1,
        )

        # TS custom hover trace
        fig.add_trace(
            go.Scatter(
                x=x_ts,
                y=DF[MdlN_S],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)'),
                text=ts_hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Parity plot
        fig.add_trace(
            go.Scatter(
                x=DF['head'],
                y=DF[MdlN_B],
                mode='markers',
                marker=dict(size=4, color=col_b),
                name=MdlN_B,
                hoverinfo='skip',
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=DF['head'],
                y=DF[MdlN_S],
                mode='markers',
                marker=dict(size=4, color=col_s),
                name=MdlN_S,
                hoverinfo='skip',
            ),
            row=1,
            col=2,
        )

        # 1:1 line
        fig.add_trace(
            go.Scatter(
                x=[adj_min, adj_max],
                y=[adj_min, adj_max],
                mode='lines',
                name='1:1',
                line=dict(color=col_11, dash='dash'),
                hoverinfo='skip',
            ),
            row=1,
            col=2,
        )

        # Parity custom hover trace, no observed marker shown
        fig.add_trace(
            go.Scatter(
                x=DF['head'],
                y=DF[MdlN_S],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)'),
                text=parity_hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Percentile plot
        fig.add_trace(
            go.Scatter(
                x=DF_Pct['Percentile'],
                y=DF_Pct['Obs'],
                mode='lines',
                name='Observed',
                line=dict(color=col_obs),
                hovertemplate=f'{"Observed:":<10} %{{y:6.2f}}<extra></extra>',
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=DF_Pct['Percentile'],
                y=DF_Pct[MdlN_B],
                mode='lines',
                name=MdlN_B,
                line=dict(color=col_b),
                hovertemplate=f'{MdlN_B + ":":<10} %{{y:6.2f}}<extra></extra>',
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=DF_Pct['Percentile'],
                y=DF_Pct[MdlN_S],
                mode='lines',
                name=MdlN_S,
                line=dict(color=col_s),
                hovertemplate=f'{MdlN_S + ":":<10} %{{y:6.2f}}<extra></extra>',
            ),
            row=2,
            col=2,
        )

        # Metrics annotation
        stats_text = (
            "<span style='font-family:Courier New; white-space:pre;'>"
            f'     {MdlN_S:>7} {MdlN_B:>7}<br>'
            f'NSE  {DF_Mtc.loc["NSE", "S"]:7.2f} {DF_Mtc.loc["NSE", "B"]:7.2f}<br>'
            f'RMSE {DF_Mtc.loc["RMSE", "S"]:7.2f} {DF_Mtc.loc["RMSE", "B"]:7.2f}<br>'
            f'MAE  {DF_Mtc.loc["MAE", "S"]:7.2f} {DF_Mtc.loc["MAE", "B"]:7.2f}<br>'
            f'Cor  {DF_Mtc.loc["Correlation", "S"]:7.2f} {DF_Mtc.loc["Correlation", "B"]:7.2f}<br>'
            f'BR   {DF_Mtc.loc["Bias Ratio", "S"]:7.2f} {DF_Mtc.loc["Bias Ratio", "B"]:7.2f}<br>'
            f'VR   {DF_Mtc.loc["Variability Ratio", "S"]:7.2f} {DF_Mtc.loc["Variability Ratio", "B"]:7.2f}<br>'
            f'KGE  {DF_Mtc.loc["KGE", "S"]:7.2f} {DF_Mtc.loc["KGE", "B"]:7.2f}'
            '</span>'
        ).format(
            DF_Mtc.loc['NSE', 'S'],
            DF_Mtc.loc['RMSE', 'S'],
            DF_Mtc.loc['MAE', 'S'],
            DF_Mtc.loc['Correlation', 'S'],
            DF_Mtc.loc['Bias Ratio', 'S'],
            DF_Mtc.loc['Variability Ratio', 'S'],
            DF_Mtc.loc['KGE', 'S'],
            DF_Mtc.loc['NSE', 'B'],
            DF_Mtc.loc['RMSE', 'B'],
            DF_Mtc.loc['MAE', 'B'],
            DF_Mtc.loc['Correlation', 'B'],
            DF_Mtc.loc['Bias Ratio', 'B'],
            DF_Mtc.loc['Variability Ratio', 'B'],
            DF_Mtc.loc['KGE', 'B'],
        )

        fig.add_annotation(
            text=stats_text,
            xref='x2 domain',
            yref='y2 domain',
            x=1,
            y=0,
            xanchor='left',
            showarrow=False,
            font=dict(size=12, family='Courier New'),
            bgcolor='white',
            borderwidth=1,
            borderpad=5,
            align='left',
        )

        # Axes
        fig.update_xaxes(title_text='Date', tickformat='%d-%b-%Y', row=1, col=1)
        fig.update_yaxes(title_text='Head (mNAP)', tickformat='.2f', row=1, col=1)
        fig.update_yaxes(title_text='Head (mNAP)', tickformat='.2f', row=2, col=2)

        tick_step = round((adj_max - adj_min) / 10, 1)
        tick_values = np.round(np.arange(adj_min, adj_max + tick_step, tick_step), 1)

        fig.update_xaxes(
            title_text='Observed Head (mNAP)',
            tickformat='.1f',
            row=1,
            col=2,
            range=[adj_min, adj_max],
            tickvals=tick_values,
        )

        fig.update_yaxes(
            title_text='Simulated Head (mNAP)',
            tickformat='.1f',
            row=1,
            col=2,
            range=[adj_min, adj_max],
            tickvals=tick_values,
        )

        fig.update_xaxes(title_text='Percentile (%)', tickformat='.1f', row=2, col=2)

        # Custom legends
        fig.add_annotation(
            text=(
                f"<b>{MdlN_S}</b> <span style='color:{col_s};'>●</span><br>"
                f"<b>{MdlN_B}</b> <span style='color:{col_b};'>●</span><br>"
                f"<b>1:1</b> <span style='color:{col_11};'>━ ━ ━</span>"
            ),
            xref='x2 domain',
            yref='y2 domain',
            x=1,
            y=1,
            xanchor='left',
            showarrow=False,
            font=dict(size=12),
            bgcolor='white',
            borderwidth=1,
            borderpad=5,
            align='left',
        )

        fig.add_annotation(
            text=(
                f"<b>{MdlN_S}</b>  <span style='color:{col_s};'>▬▬▬</span><br>"
                f"<b>{MdlN_B}</b>  <span style='color:{col_b};'>▬▬▬</span><br>"
                f"<b>Observed</b> <span style='color:{col_obs};'>▬▬▬</span>"
            ),
            xref='x3 domain',
            yref='y3 domain',
            x=1,
            y=1,
            xanchor='left',
            showarrow=False,
            font=dict(size=12),
            bgcolor='white',
            borderwidth=1,
            borderpad=5,
            align='left',
        )

        # Layout
        fig.update_layout(
            title=dict(
                text=(
                    f'<b>Groundwater Head Validation - {MdlN_S} Vs Observed Vs {MdlN_B}</b><br>'
                    f'<span style="font-size:14px; font-weight:normal;">'
                    f'Id: {Id} | X: {X}, Y: {Y} | L: {L}, R: {R}, C: {C_1}'
                    f'</span>'
                ),
                font=dict(size=20),
                y=0.98,
                x=0.5,
                xanchor='center',
            ),
            margin=dict(t=80, b=40, l=40, r=170),
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            showlegend=False,
            legend=dict(font=dict(size=10)),
            hovermode='x',
            spikedistance=1000,
            xaxis_showspikes=True,
            yaxis_showspikes=False,
            xaxis_spikemode='across',
            autosize=True,
            width=None,
            height=None,
        )

        fig.update_layout(
            margin=dict(autoexpand=True),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
        )

        sprint(f'Saving {Id:<20}', end='', indent=2)
        fig.write_html(Pa_Fo_HTML / f'{Id}.HTML', include_plotlyjs='directory')
        sprint('🟢')

    Pa_Fo_HTML_1 = M.Pa.PoP_Out_MdlN / 'GW_HD_OBS'
    Pa_Fo_HTML_2 = M.Pa.PoP_Out_MdlN / 'GW_HD_OBS/problematic'
    Pa_Fo_HTML_1.mkdir(parents=True, exist_ok=True)  # Make folder to store HTML files if it doesn't already exist.
    Pa_Fo_HTML_2.mkdir(parents=True, exist_ok=True)  # Make folder to store HTML files if it doesn't already exist.

    # %% Import and check metrics
    DF_Mtc = Vld_Mtc.to_DF()
    DF_Mtc.set_index('Metric', inplace=True)

    # %% 5.1 Make HTML Calibration plots
    # n = 8
    d_Mtc = {}
    for ID in DF_M.columns:  # [n : n + 1]:
        # Merge OBS with Mdl of S and B
        DF = DF_OBS.loc[DF_OBS['Id'] == ID].merge(
            right=DF_M[ID].rename(M.MdlN),
            how='outer',
            left_on='datetime',
            right_index=True,
        )

        if MdlN_B and MB.MdlN != M.MdlN:
            DF = DF.merge(
                right=DF_MB[ID].rename(MB.MdlN),
                how='outer',
                left_on='datetime',
                right_index=True,
            )

        DF['datetime'] = pd.to_datetime(DF['datetime'])
        DF.set_index('datetime', inplace=True)

        rule_notNA = DF[['head', M.MdlN]].notna().all(axis=1)

        if MdlN_B and MB.MdlN != M.MdlN:
            rule_notNA &= DF[MB.MdlN].notna()

        DF_notNA = DF.loc[rule_notNA]

        # Compute
        Obs = DF_notNA['head'] if not DF_notNA['head'].empty else np.nan
        S = DF_notNA[M.MdlN] if not DF_notNA[M.MdlN].empty else np.nan
        B = DF_notNA[MB.MdlN] if not DF_notNA[MB.MdlN].empty else np.nan

        DF_Mtc_I = Vld_Mtc.compute_all(
            Obs, S, B
        )  # .set_index('Metric').loc[DF_Mtc['Metric']].reset_index()  # Compute metrics for this OBS location
        d_Mtc[ID] = DF_Mtc_I['S']
        # d_Mtc[f'{ID}_B'] = DF_Mtc_I['B']

        Pctls = np.linspace(0, 100, 101)
        DF_Pct = pd.DataFrame(
            {
                'Percentile': Pctls,
                'Obs': np.percentile(Obs, Pctls),
                M.MdlN: np.percentile(S, Pctls),
                MB.MdlN: np.percentile(B, Pctls),
            }
        )

        # Info for plot
        X, Y, L, R, C_1 = DF_OBS.loc[(DF_OBS['Id'] == ID)].iloc[0][['X', 'Y', 'L', 'R', 'C']]
        min_val, max_val = (
            np.floor(min(DF['head'].min(), DF[M.MdlN].min(), DF[MB.MdlN].min()) * 10) / 10,
            np.ceil(max(DF['head'].max(), DF[M.MdlN].max(), DF[MB.MdlN].max()) * 10) / 10,
        )
        buffer = (max_val - min_val) * 0.05
        adj_min, adj_max = (min_val - buffer, max_val + buffer)
        Pa_Fo_HTML_ = (
            Pa_Fo_HTML_2 if (np.isnan(Obs).all()) or (np.isnan(S).all()) or (np.isnan(B).all()) else Pa_Fo_HTML_1
        )  # Store elsewhere if missing data

        Plot1(
            M.MdlN, MB.MdlN, DF, ID, adj_min, adj_max, DF_Pct, DF_Mtc_I, Pa_Fo_HTML_, X, Y, L, R, C_1
        )  # Create and save HTML plot for the current OBS location)

        # if (np.isnan(Obs).all()) or (np.isnan(S).all() or (np.isnan(B).all())):
        #     print(f'  X {ID} missing data or no overlap, stored in "problematic" folder.')

    DF_Mtc = pd.concat([DF_Mtc, pd.DataFrame(d_Mtc)], axis=1).copy()

    # %% 6. Create GPKG
    DF_Mtc_T = DF_Mtc.round(3).T.drop('unit')
    DF_GPKG = (
        DF_Mtc_T.merge(
            right=DF_OBS[['Id', 'X', 'Y', 'L', 'R', 'C', 'path']], how='left', left_index=True, right_on='Id'
        )
        .drop_duplicates(subset='Id')
        .dropna(subset=['NSE'])
    )
    DF_GPKG['path'] = DF_GPKG['Id'].apply(
        lambda x: f'file:///{(M.Pa.PoP_Out_MdlN / "GW_HD_OBS" / f"{x}.HTML").as_posix()}'
    )
    GDF_GPKG = gpd.GeoDataFrame(DF_GPKG, geometry=gpd.points_from_xy(DF_GPKG['X'], DF_GPKG['Y']), crs=CRS)
    metadata = {
        'MdlN': str(M.MdlN),
        'MdlN_B': str(MB.MdlN),
        'Created by': 'p_HD_OBS_TS (WS_Mdl.imod.pop.hd.py/p_HD_OBS_TS)',
        'CRS': str(CRS),
        'Description': 'Groundwater head observation point time-series plots',
        'Date': DT.now().isoformat(),
    }
    GDF_GPKG.to_file(
        M.Pa.PoP_Out_MdlN / f'GW_HD_OBS/GW_HD_OBS_Pnts_{M.MdlN}.gpkg',
        driver='GPKG',
        layer=f'GW_HD_OBS_Pnts_{M.MdlN}',
        metadata=metadata,
    )

    # %% 7. Write metadata and finish
    with open(M.Pa.PoP_Out_MdlN / 'GW_HD_OBS/metadata.txt', 'w') as f:
        f.writelines(f'{k}: {v}\n' for k, v in metadata.items())
    sprint(Sep)


def c_HD_Pctls(
    MdlN: str,
    full_years: bool = True,
    l_Pct: list | tuple = L_PCT,
    l_L: list = (1, 3, 5),  # List of layers to include in the analysis (1-based indexing)
    Pa_CSV: str | Path | None = None,
):
    """
    Calculate specified Pctls of GW HD data from a CSV file (MF6 OBS Out) and save the results as single-band TIFF files.
    """
    # %% Imports
    sprint('----- c_Pctl_HD_OBS_L initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import pandas as pd
    import rioxarray  # Noqa: F401 # activates the .rio accessor
    import xarray as xra

    sprint('🟢')

    # %% Options
    sprint('--- Loading data...')
    sprint(' -- Creating Mdl_N instance.', end='', verbose_out=False)
    M = Mdl_N(MdlN)
    Pa_CSV = list(M.Pa.MF6.glob('HD_OBS_L*.csv'))[0] if Pa_CSV is None else Pa_CSV  # Assumes only one HD_OBS_L*.csv
    sprint('🟢')

    # %%
    # DF = pd.read_csv(list(M.Pa.MF6.rglob('HD_OBS_L*.csv'))[0], engine='c', dtype='float32', sep=',' low_memory=False) # Assumes only one HD_OBS_L*.csv
    DF = timed_Exe(
        pd.read_csv,
        Pa_CSV,
        engine='c',
        dtype='float32',
        sep=',',
        low_memory=False,
        pre=f' -- Reading CSV ({Pa_CSV})...',
    )
    # DF_ = DF.copy()  # For testing

    # %% Time operations
    # DF['time'] = DT.strptime(M.SP_1st, '%Y-%m-%d') + pd.to_timedelta(DF['time'], unit='D')
    sprint(' -- Performing time operations.', end='')
    DF['time'] = M.SP_1st_DT + pd.to_timedelta(DF['time'] - 1, unit='D')

    if full_years:
        Y0 = DF['time'].dt.year.min() + (DF['time'].min() > pd.Timestamp(DF['time'].dt.year.min(), 1, 1))
        Y1 = DF['time'].dt.year.max() - (DF['time'].max() < pd.Timestamp(DF['time'].dt.year.max(), 12, 31))
        DF = DF[DF['time'].dt.year.between(Y0, Y1)]

    DF.set_index('time', inplace=True)
    sprint('🟢')

    # %% Load to DA
    sprint('--- Calculating...')
    sprint(' -- Converting to DA.', end='')

    A = DF.columns.str.extract(r'HD_(\d+)_(\d+)_(\d+)').astype('int16')
    A.columns = ['L', 'R', 'C']

    Ls = np.sort(A['L'].unique())

    l_i = pd.Index(Ls).get_indexer(A['L'])
    r_i = A['R'].to_numpy() - 1
    c_i = A['C'].to_numpy() - 1

    Arr = np.full((len(DF), len(Ls), len(M.Ys), len(M.Xs)), np.nan, dtype='float32')
    Arr[:, l_i, r_i, c_i] = DF.to_numpy(copy=False)

    DA = xra.DataArray(
        Arr,
        dims=('time', 'L', 'Y', 'X'),
        coords={
            'time': DF.index.to_numpy(),
            'L': Ls,
            'Y': M.Ys,
            'X': M.Xs,
        },
        name='HD',
    )

    sprint('🟢')

    # %% Calculate percentiles
    DA_q = timed_Exe(
        DA.quantile,
        l_Pct,
        dim='time',
        pre=' -- Calculating percentiles...',
    )

    # %% Saving to TIF
    Pa_Dir = M.Pa.PoP_Out_MdlN / 'GW_HD_Pct'
    sprint(f' -- Saving to TIF in {Pa_Dir}')
    Pa_Dir.mkdir(parents=True, exist_ok=True)
    DA_q = DA_q.rio.set_spatial_dims(x_dim='X', y_dim='Y').rio.write_crs('EPSG:28992')
    metadata = {
        'model': M.MdlN,
        'created from:': str(Pa_CSV),
        'created by': f'c_HD_Pctl_HD_OBS_L( MdlN={MdlN}, full_years={full_years}, l_Pct={l_Pct}, l_L = {l_L}, Pa_CSV = {Pa_CSV})',
    }

    for q in l_Pct:
        for L in l_L:
            Pa_Out = Pa_Dir / f'GW_HD_L{L}_{_P_label(q)}_{MdlN}.tif'

            sprint(f'  - {Pa_Out.name} ', end='')
            DA_i = DA_q.sel(quantile=q, L=L).rio.set_spatial_dims(x_dim='X', y_dim='Y')
            DA_i.rio.to_raster(
                Pa_Out,
                tags={
                    'percentile': _P_label(q),
                    'layer': str(L),
                }
                | metadata,
            )
            sprint('🟢')

    # %% 7. Write metadata and finish
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        f.writelines(f'{k}: {v}\n' for k, v in metadata.items())

    sprint('🟢🟢🟢')
    print(Sep)
    return DF, DA_q


def o_HD_Bin_Multi(
    l_MdlN: list[str] | str,
    l_Ls: list = 'all',
    Pa_Bin: list | str | Path | None = None,
    start_year: int | str = 'from_INI',
    end_year: int | str = 'from_INI',
    skip_years: int = 0,  # drop the first N years of every member (counted from start_year)
):
    """
    multi: Loads the HD_OBS_L Bin of one or more MdlNs (the members of 1 scenario) and
    concatenates them along a new 'sim' dimension. o_HD_OBS_L_Bin is called once per MdlN with
    the same arguments as before, so a single MdlN gives the same data as without this wrapper
    (with a 'sim' dimension of length 1, which does not change any statistic).

    Pa_Bin: None (resolved per MdlN), 1 path (only with 1 MdlN) or a list in the order of l_MdlN.

    Returns (DA, l_Pa_Bin), with DA dims (sim, time, layer, y, x).
    """
    import xarray as xra
    from WS_Mdl.imod.mf6.obs import o_HD_OBS_L_Bin

    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN)
    if Pa_Bin is None:
        d_Pa_Bin = dict.fromkeys(l_MdlN)
    elif isinstance(Pa_Bin, (str, Path)):
        if len(l_MdlN) > 1:
            raise TypeError('1 Pa_Bin cannot be shared by several MdlNs. Pass a list.')
        d_Pa_Bin = {l_MdlN[0]: Pa_Bin}
    else:
        d_Pa_Bin = dict(zip(l_MdlN, Pa_Bin))

    l_DA, l_Pa_Bin = [], []
    for MdlN in l_MdlN:
        M = Mdl_N(MdlN)
        Y0 = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years
        Y1 = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)

        sprint(f'  - {MdlN} ({Y0}-{Y1}) ', end='')
        l_DA.append(
            o_HD_OBS_L_Bin(
                MdlN,
                Pa_Bin=d_Pa_Bin.get(MdlN),
                l_L=l_Ls,
                start_time=M.SP_1st_DT,
                min_date=f'{Y0}-01-01',
                max_date=f'{Y1}-12-31',
            )
        )
        l_Pa_Bin.append(
            next(iter(M.Pa.MF6.rglob('HD_OBS_L*.bin')))
            if d_Pa_Bin.get(MdlN) is None
            else Path(d_Pa_Bin[MdlN])
        )  # Assumes 1 file fits the bill.
        sprint('🟢')

    # The members of 1 scenario are simulated over the same period, so the time coords must match.
    for MdlN, DA in zip(l_MdlN[1:], l_DA[1:]):
        if not np.array_equal(DA.time.values, l_DA[0].time.values):
            raise ValueError(f'{MdlN} does not have the same time coordinates as {l_MdlN[0]}.')

    return xra.concat(l_DA, dim=xra.IndexVariable('sim', l_MdlN)), l_Pa_Bin


def _ScN_Pa_Dir(M, ScN: str, sub: str):
    """multi: Out dir. Unchanged for a single MdlN, a scenario folder for several MdlNs."""
    return (M.Pa.PoP_Out_MdlN if ScN == M.MdlN else M.Pa.PoP / 'Out' / ScN) / sub


def c_HD_Bin_Pctls(  # 666 date and layer selection should be moved to the o_HD_OBS_L_Bin function. Do the same for c_HD_Bin_AVGs.
    l_MdlN: list[str] | str,  
    ScN: str | None = None,  
    full_years: bool = True,  # 666 This is not used properly.
    l_Pct: list | tuple = L_PCT,  # P01-P24, every 5 % from P25 to P75, P76-P99
    l_Ls: list = 'all',  # List of layers to include in the analysis (1-based indexing)
    Pa_Bin: list | str | Path | None = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',
    IDT: str = 'from_INI',
    skip_years: int = 0,  # drop the first N years of every member (spin-up from the starting heads)
):
    sprint('----- c_Pctl_HD_OBS_L initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', set_time=True)
    import rioxarray  # Noqa: F401 # activates the .rio accessor

    sprint('🟢', print_time=True)

    # %% Basics
    sprint('--- Loading data...', set_time=True)
    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN) # if there is one member or multiple
    M = Mdl_N(l_MdlN[0])  # the 1st member supplies the paths and the grid
    ScN = M.MdlN if ScN is None else ScN 
    start_year = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years  # skip spin-up years
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    IDT = int(M.INI.IDT) if IDT == 'from_INI' else int(IDT)
    # %% Load Bin
    DA, l_Pa_Bin = o_HD_Bin_Multi(# wraps o_HD_OBS_L_Bin, 1 call per member
        l_MdlN,
        Pa_Bin=Pa_Bin,
        l_Ls=l_Ls,
        start_year=start_year,
        end_year=end_year,
    )
    l_Ls = DA.layer.values if l_Ls == 'all' else l_Ls
    # Dask quantile needs the reduced dimension in one chunk. Spatial chunks remain bounded, so this does not combine the complete model in memory.
    DA = DA.chunk({'time': -1, 'sim': -1})
    sprint('🟢', print_time=True)

    # %% Calculate percentiles
    # .compute() once here. Otherwise the lazy dask result is recalculated for every TIF that is written below.
    DA_q = timed_Exe(
        DA.quantile(l_Pct, dim=('time', 'sim')).compute,
        pre=' -- Calculating percentiles...',
    )

    # %% Saving to TIF
    Pa_Dir = _ScN_Pa_Dir(M, ScN, 'GW_HD_Pct')  # per scenario when several members are used
    Pa_Dir.mkdir(parents=True, exist_ok=True)
    sprint(f' -- Saving to TIF in {Pa_Dir}')
    DA_q = DA_q.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:28992')

    metadata = {
        'scenario': str(ScN),  
        'model': ', '.join(l_MdlN), 
        'created from:': ' | '.join(str(p) for p in l_Pa_Bin),  # 1 path per member
        'created by': f'c_HD_Bin_Pctls(l_MdlN={l_MdlN}, ScN={ScN}, start_year={start_year}, end_year={end_year}, full_years={full_years}, IDT={IDT}, Pa_Bin={Pa_Bin}, l_Pct={l_Pct}, l_Ls={l_Ls})',
    }

    for q in l_Pct:
        for L in l_Ls:
            Pa_Out = Pa_Dir / f'L{L}/GW_HD_L{L}_{_P_label(q)}_{ScN}.tif'
            Pa_Out.parent.mkdir(parents=True, exist_ok=True)

            sprint(f'  - {Pa_Out.name} ', end='')
            DA_i = DA_q.sel(quantile=q, layer=L)
            DA_i.rio.to_raster(
                Pa_Out,
                tags={
                    'percentile': _P_label(q),
                    'layer': str(L),
                }
                | metadata,
            )
            sprint('🟢')

    # %% 7. Write metadata and finish
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        f.writelines(f'{k}: {v}\n' for k, v in metadata.items())
    sprint('🟢🟢🟢')
    print(Sep)
    return DA_q


# %%
# %%
def c_HD_Bin_AVGs(
    l_MdlN: list[str] | str,  # 1 MdlN or a list of MdlNs (the members of 1 scenario)
    ScN: str | None = None,  # scenario name used in the Out dir and file names. Defaults to the MdlN.
    full_years: bool = True,  # 666 This is not used properly.
    l_Ls: list = 'all',  # [1, 3, 5],  # List of layers to include in the analysis (1-based indexing)
    Pa_Bin: list | str | Path | None = None,  # also accepts a list of paths
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',  # inclussive
    IDT: str = 'from_INI',
    skip_years: int = 0,  # drop the first N years of every member (spin-up from the starting heads)
):
    """
    Loads Bin OBS Out data (usually all HDs per L). Calculates the AVG, summer AVG (Apr-Sep) and winter AVG over the
    whole period (all years and members pooled). Saves to TIF files. Returns the DA with all data.
    Per-year values and their percentiles: see c_HD_Bin_Yearly.
    """
    # %%
    sprint('----- c_HD_Bin_AVGs initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import rioxarray  # Noqa: F401 # activates the .rio accessor

    sprint('🟢')

    # %% Basics
    sprint('--- Loading data...')
    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN)
    M = Mdl_N(l_MdlN[0])
    ScN = M.MdlN if ScN is None else ScN
    start_year = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years  # skip spin-up years
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    IDT = int(M.INI.IDT) if IDT == 'from_INI' else int(IDT)
    # %% Load Bin
    DA, l_Pa_Bin = o_HD_Bin_Multi(
        l_MdlN,
        Pa_Bin=Pa_Bin,
        l_Ls=l_Ls,
        start_year=start_year,
        end_year=end_year,
    )
    l_Ls = DA.layer.values if l_Ls == 'all' else l_Ls
    sprint('🟢')

    # %% Saving to TIF
    Pa_Dir = _ScN_Pa_Dir(M, ScN, 'GW_HD_AVGs')
    sprint(f' -- Calculating AVGs & Saving to TIF in {Pa_Dir}')
    metadata = {
        'scenario': str(ScN),
        'model': ', '.join(l_MdlN),
        'created from:': ' | '.join(str(p) for p in l_Pa_Bin),
        'created by': f'c_HD_Bin_AVGs(l_MdlN={l_MdlN}, ScN={ScN}, start_year={start_year}, end_year={end_year}, full_years={full_years}, IDT={IDT}, Pa_Bin={Pa_Bin}, l_Ls={l_Ls})',
    }

    summer = (3 < DA.time.dt.month) & (DA.time.dt.month < 10)
    winter = ~summer
    for L in l_Ls:
        DA_L = DA.sel(layer=L)
        averages = [
            ('winter_AVG', DA_L.where(winter, drop=True).mean(dim=('time', 'sim'))),
            ('summer_AVG', DA_L.where(summer, drop=True).mean(dim=('time', 'sim'))),
            ('AVG', DA_L.mean(dim=('time', 'sim'))),
        ]

        for k, DA_i in averages:
            Pa_Out = Pa_Dir / f'L{L}/GW_HD_L{L}_{k}_{ScN}.tif'
            Pa_Out.parent.mkdir(parents=True, exist_ok=True)

            sprint(f'  - {Pa_Out.name} ', end='')
            # DA = DA.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:28992')
            DA_i.rio.to_raster(
                Pa_Out,
                tags={  # 666 add statistics with proper amount of decimals. Check how QGIS fucks them up to understand what I mean.
                    'variable': k,
                    'layer': str(L),
                }
                | metadata,
            )
            sprint('🟢')

    # %% 7. Write metadata and finish
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')
    sprint('🟢🟢🟢')
    print(Sep)
    return DA


def c_HD_Bin_Yearly(
    l_MdlN: list[str] | str,  # 1 MdlN or a list of MdlNs (the members of 1 scenario)
    ScN: str | None = None,  # scenario name used in the Out dir and file names. Defaults to the MdlN.
    l_Ls: list = 'all',  # List of layers to include in the analysis (1-based indexing)
    Pa_Bin: list | str | Path | None = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',  # inclusive
    l_Per: tuple = ('year', 'summer', 'winter'),  # calendar year, summer (Apr-Sep), winter (Oct-Mar)
    skip_years: int = 0,  # drop the first N years of every member (spin-up from the starting heads)
):
    """
    Per simulated year AND per member: the mean head of that year, of its summer (Apr-Sep) and of its winter
    (Oct-Mar). The members are different weather realisations of the same period, so year 2040 of member 1 and of
    member 2 are different years and are never averaged together: 3 members x 30 years = 90 yearly values per cell.
    Statistics over these yearly values are calculated elsewhere (e.g. c_HD_Yearly_vs_Ref).

    Winter runs from October to March, so it spans 2 calendar years. A winter is labelled with the year in which it
    starts (winter 2040 = Oct 2040 - Mar 2041). Incomplete winters at the start (only Jan-Mar of the 1st year) and at
    the end (only Oct-Dec of the last year) are dropped, so 30 simulated years give 29 winters per member.

    Saves 1 TIF per year and member, per period k ('AVG' for the year, 'summer_AVG', 'winter_AVG'):
      GW_HD_Yearly/yearly_<k>/L<L>/years/GW_HD_L<L>_<year>_<member>_<k>_<ScN>.tif
    Returns a dict {k: DA(year, sim, layer, y, x)} with the yearly values.
    """
    sprint('----- c_HD_Bin_Yearly initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import rioxarray  # Noqa: F401 # activates the .rio accessor

    sprint('🟢')

    # %% Basics
    sprint('--- Loading data...')
    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN)
    M = Mdl_N(l_MdlN[0])  # the 1st member supplies the paths and the grid
    ScN = M.MdlN if ScN is None else ScN
    start_year = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years  # skip spin-up years
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    DA, l_Pa_Bin = o_HD_Bin_Multi(
        l_MdlN,
        Pa_Bin=Pa_Bin,
        l_Ls=l_Ls,
        start_year=start_year,
        end_year=end_year,
    )
    l_Ls = list(DA.layer.values) if l_Ls == 'all' else list(l_Ls)
    sprint('🟢')

    Pa_Dir = _ScN_Pa_Dir(M, ScN, 'GW_HD_Yearly')
    metadata = {
        'scenario': str(ScN),
        'model': ', '.join(l_MdlN),
        'created from:': ' | '.join(str(p) for p in l_Pa_Bin),
        'created by': f'c_HD_Bin_Yearly(l_MdlN={l_MdlN}, ScN={ScN}, start_year={start_year}, end_year={end_year}, Pa_Bin={Pa_Bin}, l_Ls={l_Ls}, l_Per={l_Per}, skip_years={skip_years})',
    }

    summer = (3 < DA.time.dt.month) & (DA.time.dt.month < 10)
    d_Per = {}  # k -> lazy yearly means per member, dims (year, sim, layer, y, x)
    if 'year' in l_Per:
        d_Per['AVG'] = DA.groupby('time.year').mean(dim='time')
    if 'summer' in l_Per:
        d_Per['summer_AVG'] = DA.where(summer, drop=True).groupby('time.year').mean(dim='time')
    if 'winter' in l_Per:
        DA_w = DA.where(~summer, drop=True)
        w_year = (DA_w.time.dt.year - (DA_w.time.dt.month < 4).astype(int)).values  # Jan-Mar belong to the previous winter
        y0, y1 = int(DA.time.dt.year.min()), int(DA.time.dt.year.max())
        complete = (w_year >= y0) & (w_year <= y1 - 1)  # drop the incomplete 1st and last winter
        DA_w = DA_w.isel(time=np.flatnonzero(complete)).assign_coords(w_year=('time', w_year[complete]))
        d_Per['winter_AVG'] = DA_w.groupby('w_year').mean(dim='time').rename(w_year='year')

    d_Out = {}
    for k, DA_k in d_Per.items():
        # %% Yearly values per member. .load() once, so the TIFs below don't each recalculate the groupby.
        DA_Y = timed_Exe(DA_k.load, pre=f' -- Calculating yearly {k} per member...')
        DA_Y = DA_Y.transpose('year', 'sim', ...).rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:28992')

        # %% Save
        Pa_Dir_k = Pa_Dir / f'yearly_{k}'
        sprint(f' -- Saving {DA_Y.sizes["year"]} years x {DA_Y.sizes["sim"]} members to {Pa_Dir_k}')
        for L in l_Ls:
            Pa_L = Pa_Dir_k / f'L{L}/years'
            Pa_L.mkdir(parents=True, exist_ok=True)
            DA_YL = DA_Y.sel(layer=L)
            for sim in DA_YL.sim.values:
                for y in DA_YL.year.values:
                    Pa_Out = Pa_L / f'GW_HD_L{L}_{y}_{sim}_{k}_{ScN}.tif'
                    sprint(f'  - {Pa_Out.name} ', end='')
                    DA_YL.sel(sim=sim, year=y).rio.to_raster(
                        Pa_Out,
                        tags={'variable': f'yearly_{k}', 'year': str(y), 'member': str(sim), 'layer': str(L)}
                        | metadata,
                    )
                    sprint('🟢')

        d_Out[k] = DA_Y

    # %% Write metadata and finish
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        f.writelines(f'{k}: {v}\n' for k, v in metadata.items())
    sprint('🟢🟢🟢')
    print(Sep)
    return d_Out


def c_HD_Bin_Monthly(
    l_MdlN: list[str] | str,  # 1 MdlN or a list of MdlNs (the members of 1 scenario)
    ScN: str | None = None,  # scenario name used in the Out dir and file names. Defaults to the MdlN.
    l_Pct: list | tuple = L_PCT,  # P01-P24, every 5 % from P25 to P75, P76-P99
    l_Ls: list = 'all',  # [1, 3, 5], List of layers to include in the analysis (1-based indexing)
    Pa_Bin: list | str | Path | None = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',  # inclusive
    skip_years: int = 0,  # drop the first N years of every member (spin-up from the starting heads)
):
    """
    Loads Bin OBS Out data and calculates, per calendar month (1-12):
      - the mean head over all months of that calendar month (members and years pooled), 'AVG'
      - percentiles of the monthly means (1 value per member per year), 'P..'
    Saves each as a TIF per layer. Returns the monthly means per member, DA(sim, time, layer, y, x), with time = month start.
    """
    sprint('----- c_HD_Bin_Monthly initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import rioxarray  # Noqa: F401 # activates the .rio accessor

    sprint('🟢')

    # %% Basics
    sprint('--- Loading data...')
    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN)
    M = Mdl_N(l_MdlN[0])  # the 1st member supplies the paths and the grid
    ScN = M.MdlN if ScN is None else ScN
    start_year = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years  # skip spin-up years
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    DA, l_Pa_Bin = o_HD_Bin_Multi(
        l_MdlN,
        Pa_Bin=Pa_Bin,
        l_Ls=l_Ls,
        start_year=start_year,
        end_year=end_year,
    )
    l_Ls = DA.layer.values if l_Ls == 'all' else l_Ls
    sprint('🟢')
 
    # %% Monthly means per member
    DA_M = timed_Exe(
        DA.resample(time='MS').mean(dim='time').load,
        pre=' -- Calculating monthly means per member...',
    )
 
    # %% Saving to TIF
    Pa_Dir = _ScN_Pa_Dir(M, ScN, 'GW_HD_Monthly')
    sprint(f' -- Calculating monthly AVGs & percentiles & Saving to TIF in {Pa_Dir}')
    metadata = {
        'scenario': str(ScN),
        'model': ', '.join(l_MdlN),
        'created from:': ' | '.join(str(p) for p in l_Pa_Bin),
        'created by': f'c_HD_Bin_Monthly(l_MdlN={l_MdlN}, ScN={ScN}, start_year={start_year}, end_year={end_year}, Pa_Bin={Pa_Bin}, l_Pct={l_Pct}, l_Ls={l_Ls})',
    }
 
    for m in range(1, 13):
        DA_m = DA_M.isel(time=(DA_M.time.dt.month == m).values)  # 1 value per member per year
        N = int(DA_m.sizes['time'] * DA_m.sizes['sim'])
        DA_out = {'AVG': DA_m.mean(dim=('time', 'sim'))}
        DA_q = DA_m.quantile(l_Pct, dim=('time', 'sim'))
        DA_out |= {_P_label(q): DA_q.sel(quantile=q, drop=True) for q in l_Pct}

        for k, DA_k in DA_out.items():
            DA_k = DA_k.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:28992')
            for L in l_Ls:
                Pa_Out = Pa_Dir / f'L{L}/GW_HD_L{L}_M{m:02d}_{k}_{ScN}.tif'
                Pa_Out.parent.mkdir(parents=True, exist_ok=True)
 
                sprint(f'  - {Pa_Out.name} ', end='')
                DA_k.sel(layer=L).rio.to_raster(
                    Pa_Out,
                    tags={
                        'variable': k,
                        'month': str(m),
                        'layer': str(L),
                        'N monthly means': str(N),
                    }
                    | metadata,
                )
                sprint('🟢')

    # %% Write metadata and finish
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        f.writelines(f'{k}: {v}\n' for k, v in metadata.items())
    sprint('🟢🟢🟢')
    print(Sep)
    return DA_M


def c_HD_Bin_Daily(
    l_MdlN: list[str] | str,  # 1 MdlN or a list of MdlNs (the members of 1 scenario)
    ScN: str | None = None,  # scenario name used in the Out dir and file names. Defaults to the MdlN.
    l_Pct: list | tuple = L_PCT,  # P01-P24, every 5 % from P25 to P75, P76-P99
    l_Ls: list | tuple | str = (1,),  # layer 1 by default: the output is large (see the size estimate it prints)
    Pa_Bin: list | str | Path | None = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',  # inclusive
    drop_leap_day: bool = True,  # drop 29 Feb, so every calendar day has the same number of values (365 days)
    skip_years: int = 0,  # drop the first N years of every member (spin-up from the starting heads)
):
    """
    Daily climatology: per calendar day (1-365), the mean head and the percentiles over all simulated years
    and members pooled: 3 members x 30 years = 90 values per day.

    drop_leap_day=False keeps 29 Feb as day 60 of a 366-day calendar; that day then only has the leap years.

    Saves 1 TIF per calendar day and statistic: GW_HD_Daily/L<L>/<stat>/GW_HD_L<L>_D<ddd>_<stat>_<ScN>.tif,
    stat = AVG, P01, ..., P99. With the default percentiles that is 365 x 60 = 21,900 TIFs per layer, so pass a
    shorter l_Pct if you don't need them all. Each day is calculated and written before the next, so only the values
    of 1 day have to fit in memory. The number of values behind each day is in the TIF tags ('N values').
    Returns a pandas Series with that number per calendar day.
    """
    sprint('----- c_HD_Bin_Daily initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import pandas as pd
    import rioxarray  # Noqa: F401 # activates the .rio accessor

    sprint('🟢')

    # %% Basics
    sprint('--- Loading data...')
    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN)
    M = Mdl_N(l_MdlN[0])  # the 1st member supplies the paths and the grid
    ScN = M.MdlN if ScN is None else ScN
    start_year = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years  # skip spin-up years
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    DA, l_Pa_Bin = o_HD_Bin_Multi(
        l_MdlN,
        Pa_Bin=Pa_Bin,
        l_Ls=l_Ls if l_Ls == 'all' else list(l_Ls),
        start_year=start_year,
        end_year=end_year,
    )
    l_Ls = list(DA.layer.values) if l_Ls == 'all' else list(l_Ls)
    sprint('🟢')

    # %% Time checks and calendar day
    t = DA.time.to_index()
    step = pd.Series(t).diff().median()
    sprint(f' -- Output step: {step}, {DA.sizes["time"]} time steps x {DA.sizes["sim"]} members.')
    if step != pd.Timedelta(days=1):
        sprint('  ! The output step is not 1 day: each calendar day then has fewer values than expected.')

    if drop_leap_day:
        keep = ~((t.month == 2) & (t.day == 29))
        DA = DA.isel(time=np.flatnonzero(keep))
        t = t[keep]
    ref_year, n_days = (2001, 365) if drop_leap_day else (2000, 366)  # a non-leap or a leap calendar
    doy = (
        pd.to_datetime(pd.DataFrame({'year': ref_year, 'month': np.asarray(t.month), 'day': np.asarray(t.day)}))
        .dt.dayofyear.to_numpy()
    )
    days = np.arange(1, n_days + 1)

    l_stat = ['AVG'] + [_P_label(q) for q in l_Pct]
    sprint(f' -- {n_days} days x {len(l_stat)} statistics = {n_days * len(l_stat)} TIFs per layer.')

    # %% Calculate per layer and save
    Pa_Dir = _ScN_Pa_Dir(M, ScN, 'GW_HD_Daily')
    Pa_Dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'scenario': str(ScN),
        'model': ', '.join(l_MdlN),
        'created from:': ' | '.join(str(p) for p in l_Pa_Bin),
        'leap day dropped': str(drop_leap_day),
        'created by': f'c_HD_Bin_Daily(l_MdlN={l_MdlN}, ScN={ScN}, start_year={start_year}, end_year={end_year}, Pa_Bin={Pa_Bin}, l_Pct={l_Pct}, l_Ls={l_Ls}, drop_leap_day={drop_leap_day})',
    }

    d_N = {}
    for L in l_Ls:
        DA_L = DA.sel(layer=L, drop=True)
        for s in l_stat:
            (Pa_Dir / f'L{L}/{s}').mkdir(parents=True, exist_ok=True)

        sprint(f' -- L{L}: calculating and writing per day', end='')
        for d in days:
            idx = np.flatnonzero(doy == d)
            DA_d = DA_L.isel(time=idx).load()  # only the values of this day in memory
            N = int(DA_d.sizes['time'] * DA_d.sizes['sim'])
            d_N[d] = N

            DA_q = DA_d.quantile(l_Pct, dim=('time', 'sim'))
            d_out = {'AVG': DA_d.mean(dim=('time', 'sim'))} | {
                _P_label(q): DA_q.sel(quantile=q, drop=True) for q in l_Pct
            }
            for s, DA_s in d_out.items():
                DA_s.drop_vars('spatial_ref', errors='ignore').rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs(
                    'EPSG:28992'
                ).rio.to_raster(
                    Pa_Dir / f'L{L}/{s}/GW_HD_L{L}_D{d:03d}_{s}_{ScN}.tif',
                    tags={'variable': s, 'day of year': str(d), 'layer': str(L), 'N values': str(N)} | metadata,
                )
            if d % 30 == 0:
                sprint('.', end='')
        sprint(' 🟢')

    # %% Write metadata and finish
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        f.writelines(f'{k}: {v}\n' for k, v in metadata.items())
    sprint('🟢🟢🟢')
    print(Sep)
    return pd.Series(d_N, name='N values').rename_axis('day of year')


def c_HD_Bin_GXGs(
    l_MdlN: list[str] | str,
    ScN: str | None = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',
    IDT: str = 'from_INI',
    GVG: bool = False,
    l_Ls: list[int] | str = 'all',
    Pa_Bin: list | str | Path | None = None,
    skip_years: int = 0,  # drop the first N years of every member (spin-up from the starting heads)
):
    """
    Calculate GXGs from an ``HD_OBS_L`` binary and save single-band TIFs.

    The GXG calculation is the same as in ``HD_Bin_GXG_to_MBTIF``. Unlike
    that function, this function loads the continuous OBS binary with
    ``o_HD_OBS_L_Bin`` and writes a separate TIF for every variable and layer.

    with several MdlNs, calculate_gxg is run per member (it takes 1 time axis) and the
    results are combined as a mean weighted by the number of hydrological years. GHG and GLG are
    themselves means over years, so this equals the GXG over the pooled 3 x 30 years.
    """
    sprint('----- c_HD_Bin_GXGs initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False, set_time=True)
    import rioxarray  # Noqa: F401 # activates the .rio accessor
    import xarray as xra

    sprint('🟢', print_time=True)

    # %% Load and select the requested period and layers
    sprint('--- Loading data...', set_time=True)
    l_MdlN = [l_MdlN] if isinstance(l_MdlN, str) else list(l_MdlN)  
    M = Mdl_N(l_MdlN[0]) 
    ScN = M.MdlN if ScN is None else ScN 
    start_year = (M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)) + skip_years  # skip spin-up years
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    IDT = int(M.INI.IDT) if IDT == 'from_INI' else int(IDT)

    DA, l_Pa_Bin = o_HD_Bin_Multi(
        l_MdlN,
        Pa_Bin=Pa_Bin,
        l_Ls=l_Ls,
        start_year=start_year,
        end_year=end_year,
    )
    l_Ls = list(DA.layer.values)
    sprint('🟢', print_time=True)

    # %% Calculate and save GXGs
    Pa_Dir = _ScN_Pa_Dir(M, ScN, 'GXG')
    metadata = {
        'scenario': str(ScN),
        'model': ', '.join(l_MdlN),
        'created from': ' | '.join(str(p) for p in l_Pa_Bin),
        'period': f'{start_year}-{end_year}',
        'combination': 'N_years-weighted mean of the per-member GXG',
        'created by': (
            f'c_HD_Bin_GXGs(l_MdlN={l_MdlN}, ScN={ScN}, start_year={start_year}, '
            f'end_year={end_year}, IDT={IDT}, GVG={GVG}, '
            f'l_Ls={l_Ls}, Pa_Bin={Pa_Bin})'
        ),
    }
    d_GXG = {}

    sprint(f' -- Calculating GXGs & Saving to TIF in {Pa_Dir}')
    for L in l_Ls:
        Pa_Out_L = Pa_Dir / f'L{L}'
        Pa_Out_L.mkdir(parents=True, exist_ok=True)

        # multi: 1 calculate_gxg per member, then an N_years-weighted mean over the members.
        GXG_c = xra.concat(
            [imod.evaluate.calculate_gxg(DA.sel(layer=L).sel(sim=MdlN, drop=True)) for MdlN in l_MdlN],  # .load()
            dim=xra.IndexVariable('sim', l_MdlN),
        )
        GXG_c = GXG_c.rename_vars({var: var.upper() for var in GXG_c.data_vars})

        GXG = GXG_c[['GHG', 'GLG', 'GVG']].mean(dim='sim')
        GXG['N_YEARS_GXG'] = GXG_c['N_YEARS_GXG'].sum(dim='sim')
        GXG['N_YEARS_GVG'] = GXG_c['N_YEARS_GVG'].sum(dim='sim')

        N_years_GXG = np.unique(GXG.N_YEARS_GXG.values).max()
        N_years_GVG = np.unique(GXG.N_YEARS_GVG.values).max()

        GXG['GHG_m_GLG'] = GXG['GHG'] - GXG['GLG']
        variables = ['GHG', 'GLG', 'GHG_m_GLG', 'GVG'] if GVG else ['GHG', 'GLG', 'GHG_m_GLG']

        for variable in variables:
            DA_i = GXG[variable].rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:28992')
            Pa_Out = Pa_Out_L / f'{variable}_L{L}_{ScN}.tif'
            n_years = N_years_GVG if variable == 'GVG' else N_years_GXG

            sprint(f'  - {Pa_Out.name} ', end='')
            DA_i.rio.to_raster(
                Pa_Out,
                tags={
                    'variable': variable,
                    'layer': str(L),
                    'N_years': str(n_years),
                    'AVG': str(float(DA_i.mean().values)),
                }
                | metadata,
            )
            d_GXG.setdefault(variable, []).append(DA_i)
            sprint('🟢')

    d_GXG = {variable: xra.concat(arrays, dim=xra.IndexVariable('layer', l_Ls)) for variable, arrays in d_GXG.items()}
    with open(Pa_Dir / 'metadata.txt', 'w') as f:
        f.writelines(f'{key}: {value}\n' for key, value in metadata.items())

    sprint('🟢🟢🟢')
    print(Sep)
    return d_GXG
