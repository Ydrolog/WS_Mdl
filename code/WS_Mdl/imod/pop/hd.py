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
from WS_Mdl.core.text import replace_MdlN
from WS_Mdl.imod.idf import HD_Out_to_DF
from WS_Mdl.imod.prj import r_with_OBS
from WS_Mdl.xr.convert import to_TIF

__all__ = ['HD_IDF_Agg_to_TIF', 'p_HD_OBS_TS', 'c_HD_Pctls', 'HD_Pctl_Diffs', 'c_HD_Bin_Pctls', 'c_HD_Bin_AVGs']


# %%
def HD_IDF_Agg_to_TIF(
    MdlN: str,
    rules=None,
    N_cores: int = None,
    CRS: str = CRS,
    Gp: list[str] = ['year', 'month'],
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
    Gp : list of str
        Which DataFrame columns to group by. Common examples:
        - ['year','month']        → monthly aggregates
        - ['season_year','season']→ seasonal aggregates
        - ['Hy_year']             → hydrological-year aggregates
        - ['year','quarter']      → quarterly aggregates
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


def HD_Agg_name(group_keys, grouping):  # 666 could be moved to util
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
    MB = Mdl_N(MdlN_B) if isinstance(MdlN_B, str) else Mdl_N(M.B) if MdlN_B is True else M.copy()
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
        list(M.Pa.MF6.glob('HD_OBS_Pnt*.csv'))[0], index_col='time'
    )  # Read CSV file containing modelled HDs. Assumes 1 matching file
    DF_M.index = pd.to_datetime(M.SP_1st) + pd.to_timedelta(DF_M.index, unit='D')
    if MB:
        DF_MB = pd.read_csv(
            list(MB.Pa.MF6.glob('HD_OBS_Pnt*.csv'))[0], index_col='time'
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
        fig.write_html(Pa_Fo_HTML / f'{Id}.HTML')
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
        DF = DF_OBS.loc[(DF_OBS['Id'] == ID)].merge(right=DF_M[ID], how='outer', left_on='datetime', right_index=True)
        DF.rename(columns={ID: M.MdlN}, inplace=True)
        DF = DF.merge(right=DF_MB[ID].rename(index=MB.MdlN), how='outer', left_on='datetime', right_index=True)
        DF.index = pd.to_datetime(DF['datetime'])  # Set datetime as the index now that it's unique (per ID)
        DF_notNA = DF.loc[DF['head'].notna() & DF[M.MdlN].notna() & DF[MB.MdlN].notna()]

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
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')
    sprint(Sep)


def c_HD_Pctls(
    MdlN: str,
    full_years: bool = True,
    l_Pct: list = [0.05, 0.10, 0.50, 0.90, 0.95],
    l_L: list = [1, 3, 5],  # List of layers to include in the analysis (1-based indexing)
    Pa_CSV: str | Path = None,
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
            Pa_Out = Pa_Dir / f'GW_HD_L{L}_P{int(q * 100):02d}_{MdlN}.tif'

            sprint(f'  - {Pa_Out.name} ', end='')
            DA_i = DA_q.sel(quantile=q, L=L).rio.set_spatial_dims(x_dim='X', y_dim='Y')
            DA_i.rio.to_raster(
                Pa_Out,
                tags={
                    'percentile': f'P{int(q * 100):02d}',
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
    return DF, DA_q


def HD_Pctl_Diffs(MdlN_S: str, MdlN_B: str):
    """
    Calculate percentiles of the differences in GW HD between two models (S and B) and save as single-band TIFF files.
    """  # 666 needs progress reportng.
    import rioxarray

    M = Mdl_N(MdlN_S)
    MB = Mdl_N(MdlN_B)

    for F in (M.Pa.PoP_Out_MdlN / 'GW_HD_Pct').glob('GW_HD_L*_P*.tif'):
        DA_S = rioxarray.open_rasterio(F)
        Pa_B = replace_MdlN(F, M.MdlN, MB.MdlN)
        print(Pa_B)
        DA_B = rioxarray.open_rasterio(Pa_B)
        DA_Diff = DA_S - DA_B

        Pa_Out = F.parent / replace_MdlN(F.name, M.MdlN, f'{M.MdlN}m{MB.N}')
        Pa_Out.parent.mkdir(parents=True, exist_ok=True)

        print(f'Saving {Pa_Out.name} ', end='')
        DA_Diff.rio.to_raster(Pa_Out)
        print('🟢')


def c_HD_Bin_Pctls(  # 666 date and layer selection should be moved to the o_HD_OBS_L_Bin function. Do the same for c_HD_Bin_AVGs.
    MdlN: str,
    full_years: bool = True,  # 666 This is not used properly.
    l_Pct: list = [0.05, 0.10, 0.50, 0.90, 0.95],
    l_Ls: list = [1, 3, 5],  # List of layers to include in the analysis (1-based indexing)
    Pa_Bin: str | Path = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',
    IDT: str = 'from_INI',
):
    sprint('----- c_Pctl_HD_OBS_L initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import rioxarray  # Noqa: F401 # activates the .rio accessor
    from WS_Mdl.imod.mf6.obs import o_HD_OBS_L_Bin

    sprint('🟢')

    # %% Basics
    sprint('--- Loading data...', end='')
    M = Mdl_N(MdlN)
    start_year = M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    IDT = int(M.INI.IDT) if IDT == 'from_INI' else int(IDT)
    l_years = [i for i in range(start_year, end_year + 1)]

    # %% Load Bin
    DA = o_HD_OBS_L_Bin(MdlN, l_L=l_Ls, start_time=M.SP_1st_DT)
    DA = DA.where(DA.time.dt.year.isin(l_years), drop=True).sel(layer=l_Ls)  # Select specific years and layers
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
    DA_q = DA_q.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:28992')
    metadata = {
        'model': M.MdlN,
        'created from:': str(Pa_Bin),
        'created by': f'c_HD_Bin_Pctls(MdlN={MdlN}, start_year={start_year}, end_year={end_year}, full_years={full_years}, IDT={IDT}, Pa_Bin={Pa_Bin}, l_Pct={l_Pct}, l_Ls={l_Ls})',
    }

    for q in l_Pct:
        for L in l_Ls:
            Pa_Out = Pa_Dir / f'GW_HD_L{L}_P{int(q * 100):02d}_{MdlN}.tif'

            sprint(f'  - {Pa_Out.name} ', end='')
            DA_i = DA_q.sel(quantile=q, layer=L)
            DA_i.rio.to_raster(
                Pa_Out,
                tags={
                    'percentile': f'P{int(q * 100):02d}',
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
    return DA_q


# %%
def c_HD_Bin_AVGs(
    MdlN: str,
    full_years: bool = True,  # 666 This is not used properly.
    l_Ls: list = [1, 3, 5],  # List of layers to include in the analysis (1-based indexing)
    Pa_Bin: str | Path = None,
    start_year: str = 'from_INI',
    end_year: str = 'from_INI',  # inclussive
    IDT: str = 'from_INI',
):
    """
    Loads Bin OBS Out data (usually all HDs per L). Calculates summer/winter AVG, yearly AVG, etc. Saves to TIF files. Returns the DA with all data.
    """
    # %%
    sprint('----- c_Pctl_HD_OBS_L initiated -----', style=green)
    sprint('--- Loading extra packages...', end='', verbose_out=False)
    import rioxarray  # Noqa: F401 # activates the .rio accessor
    from WS_Mdl.imod.mf6.obs import o_HD_OBS_L_Bin

    sprint('🟢')

    # %% Basics
    sprint('--- Loading data...', end='')
    M = Mdl_N(MdlN)
    start_year = M.SP_1st_DT.year if start_year == 'from_INI' else int(start_year)
    end_year = M.SP_last_DT.year if end_year == 'from_INI' else int(end_year)
    IDT = int(M.INI.IDT) if IDT == 'from_INI' else int(IDT)
    l_years = [i for i in range(start_year, end_year + 1)]

    # %% Load Bin
    DA = o_HD_OBS_L_Bin(MdlN, l_L=l_Ls, start_time=M.SP_1st_DT)
    DA = DA.where(DA.time.dt.year.isin(l_years), drop=True).sel(layer=l_Ls)  # Select specific years and layers
    sprint('🟢')

    # %% Saving to TIF
    Pa_Dir = M.Pa.PoP_Out_MdlN / 'GW_HD_AVGs'
    sprint(f' -- Calculating AVGs & Saving to TIF in {Pa_Dir}')
    metadata = {
        'model': M.MdlN,
        'created from:': str(Pa_Bin),
        'created by': f'c_HD_Bin_AVGs(MdlN={MdlN}, start_year={start_year}, end_year={end_year}, full_years={full_years}, IDT={IDT}, Pa_Bin={Pa_Bin}, l_Ls={l_Ls})',
    }

    summer = (3 < DA.time.dt.month) & (DA.time.dt.month < 10)
    winter = ~summer
    for L in l_Ls:
        d_DAs = {
            'winter_AVG': DA.sel(layer=L).where(winter, drop=True).mean(dim='time'),
            'summer_AVG': DA.sel(layer=L).where(summer, drop=True).mean(dim='time'),
            'AVG': DA.sel(layer=L).mean(dim='time'),
            **{
                f'{y}_AVG': DA.sel(layer=L).where(DA.time.dt.year == y, drop=True).mean(dim='time')
                for y in np.unique(DA.time.dt.year)
            },
        }

        for k, DA_i in d_DAs.items():
            Pa_Out = (
                Pa_Dir / f'L{L}/GW_HD_L{L}_{k}_{MdlN}.tif'
                if k in ['winter_AVG', 'summer_AVG', 'AVG']
                else Pa_Dir / f'yearly_AVGs/L{L}/GW_HD_L{L}_{k}_{MdlN}.tif'
            )
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
