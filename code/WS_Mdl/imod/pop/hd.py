import os
from concurrent.futures import ProcessPoolExecutor as PPE
from datetime import datetime as DT
from pathlib import Path

import imod
from WS_Mdl.core.defaults import crs
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint
from WS_Mdl.imod.idf import HD_Out_to_DF
from WS_Mdl.xr.convert import DA_to_TIF


def HD_IDF_Agg_to_TIF(
    MdlN: str,
    rules=None,
    N_cores: int = None,
    crs: str = crs,
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
    crs : str
        Coordinate reference system for the output TIFs. By default: G.crs.
    Gp : list of str
        Which DataFrame columns to group by. Common examples:
        - ['year','month']        → monthly aggregates
        - ['season_year','season']→ seasonal aggregates
        - ['Hy_year']             → hydrological‐year aggregates
        - ['year','quarter']      → quarterly aggregates
    agg_func : str
        Name of the aggregation method to call on the xarray.DataArray (e.g. 'mean','min','max','median').
        This must exactly match a DataArray method (e.g. XA.mean(dim='time')).
    """

    def _HD_IDF_Agg_to_TIF_process(paths, Agg_F, Pa_Out, crs, params):
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

            DA_to_TIF(DA, Out, d_MtDt, crs=crs)
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
                'crs': str(crs),
                'rules': str(rules),
            }

            futures.append(
                E.submit(
                    _HD_IDF_Agg_to_TIF_process,
                    paths=list(paths),
                    Agg_F=Agg_F,
                    Pa_Out=Pa_Out,
                    crs=crs,
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
