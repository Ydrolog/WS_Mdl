from datetime import datetime as DT

import pandas as pd
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import set_verbose, warn
from WS_Mdl.imod.ini import as_d as INI_to_d


def Agg_OBS(MdlN, Pkg):

    set_verbose(False)
    M = Mdl_N(MdlN)
    start_date = INI_to_d(M.Pa.INI)['SDATE']
    l_Pa_OBS = [p for p in M.Pa.MdlN.iterdir() if f'{Pkg}_OBS' in p.name]
    set_verbose(True)

    l_DF = []

    for Pa in l_Pa_OBS:
        obs_col = Pa.name
        try:
            DF_ = pd.read_csv(Pa)
            DF_['date'] = DT.strptime(start_date, '%Y%m%d') + pd.to_timedelta(DF_['time'] - 1, unit='D')
            DF_ = DF_.drop(columns=['time'])

            Cols_no_date = [i for i in DF_.columns if i != 'date']
            DF_[obs_col] = DF_[Cols_no_date].sum(axis=1)
            DF_ = DF_[['date', obs_col]]
            l_DF.append(DF_.set_index('date')[obs_col].copy())

            print('🟢 - Successfully processed:', Pa)

        except Exception as e:
            print(f'🔴 - Failed to read {Pa}: {e}')

    if not l_DF:
        print(f'{warn}No {Pkg}_OBS files found or all failed to read for model {MdlN}.')
        return

    DF = pd.concat(l_DF, axis=1).sort_index()
    DF['SUM'] = DF.sum(axis=1, min_count=1)
    DF = DF.reset_index().rename(columns={'index': 'date'})
    Pa_Out = M.Pa.MdlN / f'OBS_Agg/{Pkg}_OBS_Agg_{MdlN}.csv'
    Pa_Out.parent.mkdir(parents=True, exist_ok=True)
    DF.to_csv(Pa_Out, index=False)

    print(f'🟢🟢 - Successfully aggregated all OBS files for {Pkg} and saved to: {Pa_Out}')
