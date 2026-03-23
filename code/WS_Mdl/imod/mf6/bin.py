from pathlib import Path

import numpy as np
import pandas as pd

from .defaults import d_Pkg_Cols


def to_DF(Pa_Bin: str | Path, Pkg: str = 'DRN') -> pd.DataFrame:
    """
    General function to read MODFLOW 6 binary input files (imod format) into a DataFrame.
    Automatically detects the correct dtype based on the package type.
    """
    # 666 I've tested it for DRN, not yet for other Pkgs. Remove this when tested for all.

    Pkg = Pkg.upper()
    if Pkg not in d_Pkg_Cols:
        raise ValueError(f'Unsupported package type: {Pkg}. Supported: {list(d_Pkg_Cols.keys())}')

    dtype = np.dtype(d_Pkg_Cols[Pkg])
    Pa_Bin = Path(Pa_Bin)

    if not Pa_Bin.exists():
        raise FileNotFoundError(f'Binary file not found: {Pa_Bin}')

    n_rec = Pa_Bin.stat().st_size // dtype.itemsize
    arr = np.fromfile(Pa_Bin, dtype=dtype, count=n_rec)

    return pd.DataFrame(arr)
