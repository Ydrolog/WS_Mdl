
"""
spinup_check.py - Spin-up check: head time series at a few points, read from the MF6 HD_OBS_L binary.
 
Why: the 2050 runs start from heads of the historic period, so the heads are too high at first and sink towards
the new equilibrium. These time series show how many years that takes, which decides SKIP_YEARS in PoP_multi.py.
 
How it reads the 77 GB file without loading it:
    The MF6 observation binary is: a 100-character header, the number of observations, their names, and then per
    time step 1 time value followed by 1 value per observation. The file is opened as a memory map, and per time step
    only the columns of the requested points are read. The reading time therefore depends on the number of points
    and time steps, not on the size of the model.
 
Points CSV (1 row per point; column names are not case-sensitive)
    ID, L, X, Y
 
1 VM = 1 scenario: set SCENARIOS to the members on this VM

Outputs in <PoP>/Out/SpinUp/<scenario>/:
    TS_daily.csv            daily head per point and member (long format: scenario, member, ID, date, head)
    TS_annual.csv           annual mean head per point, member and simulated year (year 1 = first year of the run)
    TS_points.html          per point: daily head (thin) and 365-day running mean (thick), per member (interactive)
    TS_annual_anomaly.html  annual mean minus the mean of the last N_LAST years, per point (thin) and median (thick)
 
Run from the pixi shell:  python -u spinup_check.py
"""
 
import math
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)
 
# --- Input -----------------------------------------------------------------------------------------------------------
CSV_POINTS = r'G:\code\WS_Mdl\imod\pop\spinup_points.csv' 
SCENARIOS = {  
    'Hd_2050': ['NBr131', 'NBr132', 'NBr133'],
}
N_YEARS = 10  # first N years of every run
N_LAST = 3  # the last N_LAST years read are the comparison level in the anomaly figure
PA_BIN = {}  
# ---------------------------------------------------------------------------------------------------------------------
def r_OBS_Bin_header(Pa_Bin):
    """Reads the header of an MF6 observation binary. Returns a dict with the layout of the data part."""
    Pa_Bin = Path(Pa_Bin)
    with open(Pa_Bin, 'rb') as f:
        cline = f.read(100).decode('ascii', errors='replace')
        if not cline.startswith('cont'):
            raise ValueError(f'{Pa_Bin} does not look like an MF6 observation binary (header: {cline[:20]!r}).')
        precision = 'double' if 'double' in cline[5:11].lower() else 'single'
        lenobsname = int(cline[11:])
        nobs = int(np.fromfile(f, dtype='<i4', count=1)[0])
        names = np.frombuffer(f.read(nobs * lenobsname), dtype=f'S{lenobsname}')
        data_start = f.tell()

    fsize = 8 if precision == 'double' else 4
    rec_bytes = (nobs + 1) * fsize  # 1 time value + nobs values per time step
    n_bytes = Pa_Bin.stat().st_size - data_start
    if n_bytes % rec_bytes:
        print(f'  ! {Pa_Bin.name}: the last time step is incomplete (file still being written?); it is ignored.')
    return {
        'path': Pa_Bin,
        'dtype': f'<f{fsize}',
        'nobs': nobs,
        'names': names,
        'data_start': data_start,
        'n_steps': n_bytes // rec_bytes,
    }


def find_columns(header, l_name):
    """Column index (0-based, without the time column) of each observation name. Case and padding are ignored."""
    names = np.char.upper(np.char.strip(header['names']))
    wanted = np.array([n.upper().encode() for n in l_name])
    order = np.argsort(names)
    pos = np.searchsorted(names, wanted, sorter=order)
    pos = np.clip(pos, 0, len(names) - 1)
    idx = order[pos]
    found = names[idx] == wanted
    if not found.all():
        missing = [n for n, ok in zip(l_name, found) if not ok]
        examples = [n.decode() for n in names[:5]]
        raise KeyError(
            f'🔴 - {len(missing)} point(s) not in {header["path"].name}: {missing[:5]}\n'
            f'     Names in the file look like: {examples}. Check the L/R/C of the points (1-based?).'
        )
    return idx


def r_OBS_Bin_points(header, l_name, start_date, n_years):
    """Reads the given observations for the first n_years. Returns a DataFrame (index = date, columns = names)."""
    idx = find_columns(header, l_name)
    mm = np.memmap(
        header['path'],
        dtype=header['dtype'],
        mode='r',
        offset=header['data_start'],
        shape=(header['n_steps'], header['nobs'] + 1),
    )
    t = np.asarray(mm[:, 0], dtype='float64')  # time in days since the start of the run (end of each time step)
    dates = pd.Timestamp(start_date) + pd.to_timedelta(t - 1, unit='D')  # same convention as c_HD_Pctls
    end = pd.Timestamp(start_date) + pd.DateOffset(years=n_years)
    n = int(np.searchsorted(dates.values, end.to_datetime64()))
    if n == 0:
        raise ValueError(f'No time steps before {end.date()} in {header["path"].name}.')

    order = np.argsort(idx)  # read the columns from left to right, then put them back in the CSV order
    cols = idx[order] + 1  # +1 for the time column
    vals = np.empty((n, len(idx)), dtype='float64')
    vals[:, order] = mm[:n, cols]
    del mm
    return pd.DataFrame(vals, index=pd.DatetimeIndex(dates[:n], name='date'), columns=l_name)


def points_to_names(DF_pts, M):
    """Adds L, R, C and the observation name HD_<L>_<R>_<C> to the points table."""
    DF = DF_pts.copy()
    DF.columns = [c.strip() for c in DF.columns]
    DF = DF.rename(columns={c: c.upper() for c in DF.columns if c.upper() in ('ID', 'L', 'X', 'Y', 'R', 'C')})
    if 'ID' not in DF.columns:
        DF['ID'] = [f'P{i + 1:02d}' for i in range(len(DF))]
    if not {'R', 'C'}.issubset(DF.columns):
        if not {'X', 'Y'}.issubset(DF.columns):
            raise KeyError('The points CSV needs either X, Y or R, C columns.')
        import WS_Mdl.core.df  # noqa: F401  # registers the .ws accessor (as in p_HD_OBS_TS)

        DF = DF.ws.XY_to_RC(M, x='X', y='Y')
    DF[['L', 'R', 'C']] = DF[['L', 'R', 'C']].astype(int)
    DF['name'] = [f'HD_{L}_{R}_{C}' for L, R, C in DF[['L', 'R', 'C']].itertuples(index=False)]
    return DF


def _colors(DF):
    """1 colour per (scenario, member): oranges for the 1st scenario, blues for the 2nd, etc."""
    palettes = [
        ['#FDBA74', '#F97316', '#C2410C', '#7C2D12'],
        ['#93C5FD', '#3B82F6', '#1D4ED8', '#1E3A8A'],
        ['#86EFAC', '#22C55E', '#15803D', '#14532D'],
        ['#D8B4FE', '#A855F7', '#7E22CE', '#581C87'],
    ]
    out = {}
    for pal, (ScN, DS) in zip(palettes, DF.groupby('scenario', sort=False)):
        for i, m in enumerate(DS['member'].unique()):
            out[(ScN, m)] = pal[i % len(pal)]
    return out


def p_points(DF_daily, DF_pts, Pa_Out):
    """Per point: daily head (thin) and 365-day running mean (thick) for every member. Interactive HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    l_ID = list(DF_pts['ID'])
    ncols = 4
    nrows = math.ceil(len(l_ID) / ncols)
    titles = [f'{r.ID} (L{r.L} R{r.R} C{r.C})' for r in DF_pts.itertuples()]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, shared_xaxes=True,
                        vertical_spacing=0.06 if nrows > 3 else 0.1, horizontal_spacing=0.05)
    colors = _colors(DF_daily)

    for i, ID in enumerate(l_ID):
        row, col = i // ncols + 1, i % ncols + 1
        D = DF_daily[DF_daily['ID'] == ID]
        for (ScN, member), DM in D.groupby(['scenario', 'member'], sort=False):
            s = DM.set_index('date')['head']
            x = (s.index - s.index[0]).days / 365.25  # years since the start of the run, so scenarios overlap
            name, c = f'{ScN} {member}', colors[(ScN, member)]
            hover = s.index.strftime('%d-%b-%Y')
            fig.add_trace(
                go.Scattergl(x=x, y=s.values, mode='lines', line=dict(width=0.8, color=c), opacity=0.35,
                             name=name, legendgroup=name, showlegend=False, customdata=hover,
                             hovertemplate=f'{ID} {name}<br>%{{customdata}}: %{{y:.2f}} m NAP<extra></extra>'),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scattergl(x=x, y=s.rolling(365, center=True).mean().values, mode='lines',
                             line=dict(width=2.5, color=c), name=name, legendgroup=name, showlegend=(i == 0),
                             hovertemplate=f'{ID} {name}<br>year %{{x:.1f}}, 365-d mean: %{{y:.2f}} m NAP<extra></extra>'),
                row=row, col=col,
            )
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text='Years since the start of the run', row=nrows, col=c)
    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text='Head (m NAP)', row=r, col=1)
    fig.update_layout(
        title='Head at the spin-up points: daily (thin) and 365-day running mean (thick). Click a legend item to hide it.',
        height=300 * nrows + 120, template='plotly_white', hovermode='closest',
        legend=dict(orientation='h', y=-0.08 if nrows > 3 else -0.15, x=0),
    )
    fig.write_html(Pa_Out, include_plotlyjs='directory')


def p_annual_anomaly(DF_annual, Pa_Out, n_last):
    """Annual mean minus the mean of the last n_last years read, per point and member, with the median per scenario."""
    import plotly.graph_objects as go

    D = DF_annual.copy()
    last = D['year'] > D.groupby(['scenario', 'member'])['year'].transform('max') - n_last
    level = D[last].groupby(['scenario', 'member', 'ID'])['head'].mean().rename('level')
    D = D.join(level, on=['scenario', 'member', 'ID'])
    D['anomaly'] = D['head'] - D['level']

    fig = go.Figure()
    base = {ScN: c for ScN, c in zip(D['scenario'].unique(), ['#C2410C', '#1D4ED8', '#15803D', '#7E22CE'])}
    for (ScN, member, ID), DP in D.groupby(['scenario', 'member', 'ID'], sort=False):
        fig.add_trace(go.Scatter(
            x=DP['year'], y=DP['anomaly'], mode='lines', line=dict(width=1, color=base[ScN]), opacity=0.25,
            legendgroup=f'{ScN} points', name=f'{ScN}: single points', showlegend=False,
            hovertemplate=f'{ScN} {member} {ID}<br>year %{{x}}: %{{y:.3f}} m<extra></extra>',
        ))
    for ScN, DS in D.groupby('scenario', sort=False):
        med = DS.groupby('year')['anomaly'].median()
        fig.add_trace(go.Scatter(
            x=med.index, y=med.values, mode='lines+markers', line=dict(width=4, color=base[ScN]),
            name=f'{ScN}: median of all points and members',
            hovertemplate=f'{ScN} median<br>year %{{x}}: %{{y:.3f}} m<extra></extra>',
        ))
    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.update_layout(
        title='Spin-up: where does the annual mean head stop sinking?',
        xaxis=dict(title='Simulated year (1 = first year of the run)', dtick=1),
        yaxis=dict(title=f'Annual mean head minus mean of the last {n_last} years (m)'),
        template='plotly_white', height=600, hovermode='closest',
    )
    fig.write_html(Pa_Out, include_plotlyjs='directory')


def main():
    from WS_Mdl.core.mdl import Mdl_N

    DF_pts_in = pd.read_csv(CSV_POINTS)
    M0 = Mdl_N(next(iter(SCENARIOS.values()))[0])
    DF_pts = points_to_names(DF_pts_in, M0)
    print(f'--- {len(DF_pts)} points, first {N_YEARS} years:')
    print(DF_pts[['ID', 'L', 'R', 'C', 'name']].to_string(index=False))

    l_daily = []
    for ScN, l_MdlN in SCENARIOS.items():
        for MdlN in l_MdlN:
            M = Mdl_N(MdlN)
            Pa_Bin = Path(PA_BIN[MdlN]) if MdlN in PA_BIN else next(M.Pa.MF6.rglob('HD_OBS_L*.bin'), None)
            if Pa_Bin is None or not Pa_Bin.exists():
                raise FileNotFoundError(f'🔴 - No HD_OBS_L*.bin for {MdlN} in {M.Pa.MF6}. Set it in PA_BIN.')
            print(f'--- {ScN} {MdlN}: {Pa_Bin}')
            header = r_OBS_Bin_header(Pa_Bin)
            print(f'    {header["nobs"]:,} observations, {header["n_steps"]:,} time steps, {header["dtype"]}')
            DF = r_OBS_Bin_points(header, list(DF_pts['name']), M.SP_1st_DT, N_YEARS)
            DF.columns = list(DF_pts['ID'])
            print(f'    read {DF.index[0].date()} - {DF.index[-1].date()} ({len(DF):,} days)')
            l_daily.append(
                DF.reset_index().melt(id_vars='date', var_name='ID', value_name='head').assign(scenario=ScN, member=MdlN)
            )

    DF_daily = pd.concat(l_daily, ignore_index=True)[['scenario', 'member', 'ID', 'date', 'head']]
    Pa_Out = M0.Pa.PoP / 'Out' / 'SpinUp' / '_'.join(SCENARIOS)
    Pa_Out.mkdir(parents=True, exist_ok=True)
    DF_daily.to_csv(Pa_Out / 'TS_daily.csv', index=False)

    first_year = DF_daily.groupby('member')['date'].transform('min').dt.year
    DF_daily['year'] = DF_daily['date'].dt.year - first_year + 1
    DF_annual = DF_daily.groupby(['scenario', 'member', 'ID', 'year'], as_index=False)['head'].mean()

    DF_annual.to_csv(Pa_Out / 'TS_annual.csv', index=False)
    p_points(DF_daily, DF_pts, Pa_Out / 'TS_points.html')
    p_annual_anomaly(DF_annual, Pa_Out / 'TS_annual_anomaly.html', N_LAST)
    print(f'🟢 Saved to {Pa_Out}')


if __name__ == '__main__':
    main()