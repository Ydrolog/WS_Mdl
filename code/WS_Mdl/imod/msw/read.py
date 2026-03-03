import json
from pathlib import Path

import pandas as pd
from WS_Mdl.core.path import Pa_WS
from WS_Mdl.core.style import sprint


def MSW_In_to_DF(Pa: Path | str) -> pd.DataFrame:
    """
    Converts the contents of a MSW input file into a pandas DataFrame.
    """
    Pa = Path(Pa)
    Fi = Pa.name

    d_headers = json.load(open(Path(Pa_WS) / 'code/WS_Mdl/Auxi/MSW_headers.json'))
    d_colspecs = json.load(open(Path(Pa_WS) / 'code/WS_Mdl/Auxi/MSW_colspecs.json'))

    try:
        if Fi == 'mete_grid.inp':
            DF = pd.read_csv(Pa, header=None, names=d_headers[Fi])
        elif Fi in d_colspecs:
            DF = pd.read_fwf(Pa, colspecs=d_colspecs[Fi], header=None)  # , l_headers=d_headers[Fi])
            DF.columns = d_headers[Fi][: DF.shape[1]]
        elif Fi == 'para_sim.inp':
            DF = pd.DataFrame({'Line': [ln.strip() for ln in open(Pa) if '=' in ln]})
            DF['parameter'] = DF['Line'].apply(lambda x: x.split('=')[0].strip())
            DF['value'] = DF['Line'].apply(lambda x: x.split('=')[1].split('!')[0].strip() if '=' in x else '')
            DF['comment'] = DF['Line'].apply(lambda x: x.split('!')[1].strip() if ('=' in x) and ('!' in x) else '')
            DF.drop(columns=['Line'], inplace=True)
        elif Fi == 'sel_key_svat_per.inp':
            DF = pd.DataFrame({'Line': [ln.strip() for ln in open(Pa)]})
            DF['parameter'] = DF['Line'].apply(lambda x: x.split()[0])
            DF['value'] = DF['Line'].apply(lambda x: x.split()[1])
            DF.drop(columns=['Line'], inplace=True)
        else:
            DF = pd.read_fwf(Pa, header=None)  # , l_headers=d_headers[i]
            DF.columns = d_headers[Fi][: DF.shape[1]]
        sprint(Fi, '🟢')
        return DF
    except Exception as e:
        sprint(Fi, '🔴', e)
        return
