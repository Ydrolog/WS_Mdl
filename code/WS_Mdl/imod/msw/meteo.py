from pathlib import Path

import imod
import xarray as xra
from tqdm import tqdm


def to_XA(DF_meteo, Par, Pa_PRJ, Xmin=None, Ymin=None, Xmax=None, Ymax=None):
    """
    Reads multiple .asc files listed in one of DF_meteo's columns and combines them into a single xarray DataArray.
    """

    base_dir = Path(Pa_PRJ).parent

    l_da = []
    # Iterate over the DataFrame rows to read each .asc file # I've tried parallelizing this, and the speed was about the same. So I'm keeping the simpler serial approach.
    for index, row in tqdm(DF_meteo.iterrows(), total=len(DF_meteo), desc=f' -- Loading {Par}'):
        file_path = (base_dir / row[Par]).resolve()

        # Read the .asc file using imod.rasterio (lazy)
        da = imod.rasterio.open(file_path)
        if 'band' in da.dims:
            da = da.squeeze('band', drop=True)

        # Sort by y to ensure slicing works correctly (rasters are often descending y)
        da = da.sortby('y')

        # Select only the Area of Interest
        da = da.sel(x=slice(Xmin, Xmax), y=slice(Ymin, Ymax))

        l_da.append(da)

    if l_da:
        # Concatenate along the 'time' dimension directly using the datetime values
        # This keeps 'time' as a single dimension instead of MultiIndex (year, day)
        # We use 'DT' column which we prepared earlier
        times = DF_meteo['DT'].values
        A_P = xra.concat(l_da, dim='time')
        A_P = A_P.assign_coords(time=times)
    else:
        print('No data loaded.')
        A_P = None
    return A_P
