from datetime import datetime as DT

import pandas as pd
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import set_verbose, sprint, warn


def Agg_OBS(MdlN, Pkg, save=True, overwrite=False, MF6_NaN_to_zero=True):

    set_verbose(False)
    M = Mdl_N(MdlN)
    start_date = M.INI.SDATE
    start_date = DT.strptime(start_date, '%Y%m%d')
    l_Pa_OBS = list(M.Pa.MdlN.rglob(f'{Pkg}_OBS*.CSV'))
    set_verbose(True)

    l_DF = []

    for Pa in l_Pa_OBS:
        obs_col = Pa.name
        try:
            DF_ = pd.read_csv(Pa)
            if MF6_NaN_to_zero:
                DF_ = DF_.replace(3.0e30, 0)
            DF_['date'] = start_date + pd.to_timedelta(DF_['time'] - 1, unit='D')
            DF_ = DF_.drop(columns=['time'])

            Cols_no_date = [i for i in DF_.columns if i != 'date']
            DF_[obs_col] = DF_[Cols_no_date].sum(axis=1)
            DF_ = DF_[['date', obs_col]]
            l_DF.append(DF_.set_index('date')[obs_col].copy())

            sprint('🟢 - Successfully processed:', Pa)

        except Exception as e:
            sprint(f'🔴 - Failed to read {Pa}: {e}')

    if not l_DF:
        sprint(f'{warn}No {Pkg}_OBS files found or all failed to read for model {MdlN}.')
        return

    DF = pd.concat(l_DF, axis=1).sort_index()
    DF['SUM'] = DF.sum(axis=1, min_count=1)
    DF = DF.reset_index().rename(columns={'index': 'date'})
    Pa_Out = M.Pa.MF6 / f'OBS_Agg/{Pkg}_OBS_Agg_{MdlN}.csv'

    if save:
        Pa_Out.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not Pa_Out.exists():
            DF.to_csv(Pa_Out, index=False)

    sprint(f'🟢🟢 - Successfully aggregated all OBS files for {Pkg} and saved to: {Pa_Out}')
    return DF
