"""
Read MSW input files (.inp).

Functions for each file specifically, are provided in a .py file in the same folder (e.g., mete_grid.py for mete_grid.inp).
"""

from functools import cached_property
from pathlib import Path

import pandas as pd
import xarray as xra
import yaml
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import sprint


def to_DF(Pa: Path | str) -> pd.DataFrame:
    """
    Converts the contents of a MSW input file into a pandas DataFrame.
    """
    Pa = Path(Pa)
    Fi = Pa.name.lower()

    # Get headers and colspecs, as in MSW documentation. Necessary cause the files have no headers and fixed width columns.
    d_headers = yaml.safe_load(open(Path(__file__).parent / 'defaults/headers.yaml'))
    d_colspecs = yaml.safe_load(open(Path(__file__).parent / 'defaults/colspecs.yaml'))

    try:
        status = '🟢'
        if Fi == 'mete_grid.inp':
            DF = pd.read_csv(Pa, header=None, names=d_headers[Fi])
        elif Fi in d_colspecs:
            DF = pd.read_fwf(Pa, colspecs=d_colspecs[Fi], header=None)  # , l_headers=d_headers[Fi])
            if DF.shape[1] > len(d_headers[Fi]):
                status = f'🟡 - Missing {DF.shape[1] - len(d_headers[Fi])} columns'
                DF = DF.iloc[:, : len(d_headers[Fi])]
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
            DF = pd.read_csv(Pa, header=None, sep=r'\s+')
            if DF.shape[1] > len(d_headers[Fi]):
                status = f'🟡 - Missing {DF.shape[1] - len(d_headers[Fi])} columns'
                DF = DF.iloc[:, : len(d_headers[Fi])]
            DF.columns = d_headers[Fi][: DF.shape[1]]
        sprint(Fi, status)
        return DF
    except Exception as e:
        sprint(Fi, '🔴', e)
        return


class MSW_In:
    """
    Class for reading ALL MSW input files.
    Should run fast, but if you want to read just one file faster, use the to_DF function directly with the path to that file.
    """

    def __init__(self, M: Mdl_N):
        if not isinstance(M, Mdl_N):
            raise TypeError(f'MdlN expected for MSW_In initialization, got {type(M)}')
        self.Pa = sorted(M.Pa.MSW.glob('*.inp'))

    @cached_property
    def d(self) -> dict[str, pd.DataFrame | None]:
        """
        Returns a dictionary of pandas DataFrames from the MSW input files.
        """

        d_DF = {}

        for i, Pa_Inp in enumerate(self.Pa, start=1):
            print(f'{i:2}', end=' - ')
            DF = to_DF(Pa_Inp)
            d_DF[Pa_Inp.stem.lower()] = DF.rename(columns={'SVAT unit': 'SVAT'}) if isinstance(DF, pd.DataFrame) else DF

        return d_DF

    @cached_property
    def DS(self) -> xra.Dataset:
        """Prototype: build and cache an xarray Dataset from d_MSW_In output."""

        # Initiate DS from idf_svat
        DF_SVAT = self.d['idf_svat']
        DF_SVAT.rename(
            columns={'row number of gridcell (MODFLOW style)': 'R', 'column number of gridcell (MODFLOW style)': 'C'},
            inplace=True,
        )
        DF_SVAT['N'] = DF_SVAT.groupby(
            ['R', 'C']
        ).cumcount()  # 3rd dimesion, to be able to store multiple SVATs for the same R C to the DS
        DS = DF_SVAT.set_index(['R', 'C', 'N']).to_xarray()

        for i, Fi in enumerate(
            [i for i in self.d.keys() if i != 'idf_svat'], start=1
        ):  # Loop through all files except for idf_svat.
            print(f'{i:2}', end=' - ')
            try:
                DF_ = self.d[Fi].set_index('SVAT')

                for Col in DF_.columns:
                    DS[Col] = (
                        DS['SVAT'].dims,
                        DF_[Col].reindex(DS['SVAT'].values.ravel()).to_numpy().reshape(DS['SVAT'].shape),
                    )
                sprint(f'{Fi} 🟢')
            except Exception as e:
                sprint(f'{Fi} 🔴', e)

        return DS
