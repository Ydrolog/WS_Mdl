from pathlib import Path

import numpy as np
import pandas as pd


def to_DF(Pa_Bin: str | Path, Pkg: str = 'DRN') -> pd.DataFrame:
    """
    General function to read MODFLOW 6 binary input files (imod format) into a DataFrame.
    Automatically detects the correct dtype based on the package type.
    """
    # 666 I've tested it for DRN, not yet for other Pkgs. Remove this when tested for all.

    # Mapping of package types to their MF6 binary structure
    d_Dtypes = {
        'DRN': [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('elev', '<f8'), ('cond', '<f8')],
        'RIV': [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('stage', '<f8'), ('cond', '<f8'), ('rbot', '<f8')],
        'GHB': [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('bhead', '<f8'), ('cond', '<f8')],
        'WEL': [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('q', '<f8')],
        'CHD': [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('head', '<f8')],
    }

    Pkg = Pkg.upper()
    if Pkg not in d_Dtypes:
        raise ValueError(f'Unsupported package type: {Pkg}. Supported: {list(d_Dtypes.keys())}')

    dtype = np.dtype(d_Dtypes[Pkg])
    Pa_Bin = Path(Pa_Bin)

    if not Pa_Bin.exists():
        raise FileNotFoundError(f'Binary file not found: {Pa_Bin}')

    n_rec = Pa_Bin.stat().st_size // dtype.itemsize
    arr = np.fromfile(Pa_Bin, dtype=dtype, count=n_rec)

    return pd.DataFrame(arr)
