"""Generate a MODFLOW 6 MVR package from the *new-style* mapping **DF**.

The mapping **DF** must come from ``match_cells_to_SFR`` in
``DRN_SFR_match.py``.  Column names and conventions expected are :

* **Tgt_L, Tgt_R, Tgt_C**  -- 1-based layer/row/column of the **source**
  (DRN) cell.
* **SFR_L, SFR_R, SFR_C** -- 1-based layer/row/column of the **receiver**
  (SFR) cell.
* **distance** -- ignored by the writer but allowed to be present.

Because the indices are already 1-based, *no* additional offset is
applied here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
from WS_Mdl import utils as U
import pandas as pd

__all__: list[str] = ['w_MVR']

# Helper / validation -----------------------------------------------------------------------------
_EXPECTED_COLS = ['Pvd_i', 'Rcv_i']


def _validate_df(df: pd.DataFrame) -> None:
    """Ensure required columns are present; raise ValueError otherwise."""
    missing = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError('Input DataFrame is missing required column(s): ' + ', '.join(missing))


# Public writer -----------------------------------------------------------------------------
def w_MVR(
    DF: pd.DataFrame,
    *,
    filename: Union[str, Path] = 'DRN_to_SFR.mvr',
    factor_value: float | int = 1,
    period: int = 1,
    Pkg1: str = 'DRN',
    Pkg2: str = 'SFR',
) -> Path:
    """Write a single-period MVR package file using the **new** column names.

    Parameters
    ----------
    DF
        DataFrame returned by ``match_cells_to_SFR``.
    source_pkg, receiver_pkg
        Package names as used in the MODFLOW 6 name file.
    filename
        Output path (created if necessary).
    factor_value
        ``FACTOR`` assigned to *every* connection.
    period
        Stress-period number (1-based).

    Returns
    -------
    pathlib.Path
        Path of the written MVR file.
    """
    _validate_df(DF)

    DF['Pkg1'] = Pkg1
    DF['Pkg2'] = Pkg2
    DF['MVR_type'] = 'FACTOR'
    DF['value'] = 1
    DF = DF[['Pkg1', 'Pvd_i', 'Pkg2', 'Rcv_i', 'MVR_type', 'value']]

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Build file l
    l: list[str] = []
    l.append('BEGIN OPTIONS')
    l.append('END OPTIONS')
    l.append('')

    l.append('BEGIN DIMENSIONS')
    l.append(f'\tMAXMVR {DF.shape[0]}')
    l.append('\tMAXPACKAGES 2')  # 666 If we add more packages, this will need to be updated.
    l.append('END DIMENSIONS')
    l.append('')

    l.append('BEGIN PACKAGES')
    l.append(f'  {Pkg1}')
    l.append(f'  {Pkg2}')
    l.append('END PACKAGES')
    l.append('')

    l.append(f'BEGIN PERIOD {period}')

    # for _, r in DF.iterrows():
    #     line = (f"  {source_pkg} {int(r.Tgt_L)} {int(r.Tgt_R)} {int(r.Tgt_C)}   "
    #             f"{receiver_pkg} {int(r.SFR_L)} {int(r.SFR_R)} {int(r.SFR_C)}   "
    #             f"FACTOR {factor_value}")
    l.append(U.DF_to_MF_block(DF))

    l.append('END PERIOD')
    l.append('')

    filename.write_text('\n'.join(l))
    return filename


if __name__ == '__main__':  # pragma: no cover
    # Quick smoke test using a tiny dummy DF
    import numpy as np

    dummy = pd.DataFrame(
        {
            'Pvd_i': [1, 1],
            'Rcv_i': [2, 3],
        }
    )

    out = w_MVR(dummy, filename='_test_drn_to_sfr.mvr')
    print('Wrote', out)
