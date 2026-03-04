from pathlib import Path

import xarray as xra

from WS_Mdl.core.style import set_verbose, sprint
from WS_Mdl.xr.convert import to_MBTIF


def Diff_MBTIF(Pa_TIF1, Pa_TIF2, Pa_TIF_Out=None, verbose=True):
    """
    Calculates the difference between two Multi-band TIF files (TIF1 - TIF2).
    Assumes both files have the same number of bands and dimensions.
    """

    Pa_TIF1, Pa_TIF2 = Path(Pa_TIF1), Path(Pa_TIF2)

    if Pa_TIF_Out is None:
        Pa_TIF_Out = Pa_TIF1.parent / f'{Pa_TIF1.stem}_diff.tif'

    if verbose:
        sprint(f'Calculating difference: {Pa_TIF1.name} - {Pa_TIF2.name}')

    ds1 = xra.open_dataset(Pa_TIF1, engine='rasterio')
    ds2 = xra.open_dataset(Pa_TIF2, engine='rasterio')
    try:
        da1 = ds1['band_data']
        da2 = ds2['band_data']

        if da1.shape != da2.shape:
            raise ValueError(f'Shapes do not match: {da1.shape} vs {da2.shape}')

        da_diff = da1 - da2

        d_MtDt = {f'Diff_Band_{i + 1}': {'description': f'Difference Band {i + 1}'} for i in range(da_diff.shape[0])}

        set_verbose(False)
        try:
            to_MBTIF(da_diff, Pa_TIF_Out, d_MtDt, _print=verbose)
        finally:
            set_verbose(True)

        if verbose:
            sprint(f'🟢 - Difference saved to {Pa_TIF_Out}')
    finally:
        ds1.close()
        ds2.close()
