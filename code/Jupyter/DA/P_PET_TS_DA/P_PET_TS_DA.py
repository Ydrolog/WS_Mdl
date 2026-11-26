# %%
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from WS_Mdl.core.mdl import Mdl_N

# %% [markdown]
# # Define functions


# %%
def _read_asc_grid(file_path, dtype, window=None):
    # Each worker opens its own dataset. Avoid scanning the large source directory.
    gdal_dtype = 'Float32' if dtype == np.dtype('float32') else 'Float64'
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='TRUE', AAIGRID_DATATYPE=gdal_dtype):
        with rasterio.open(file_path) as src:
            data = src.read(1, window=window, out_dtype=dtype.name)
            grid = (src.shape, src.transform, src.crs)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
    return data, grid


def r_asc(file_path, dtype=np.float32):
    """Read an ASCII raster with GDAL, replacing NoData with NaN."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype must be float32 or float64.')
    return _read_asc_grid(file_path, dtype)[0]


# %%
def combine_asc_files(
    folder_path, file_names, *, aggregate=None, name=None, max_workers=16, dtype=np.float32, bounds=None
):
    """Read dated ASC files into an xarray DataArray using bounded parallel reads.

    Filenames must contain a YYYYMMDD field, e.g. P_19910101_NBr1.asc.
    aggregate=None returns (time, y, x); aggregate='year' returns (year, y, x)
    without keeping daily grids in memory. Annual sums use float64 accumulators,
    skip NoData, and retain NaN where a cell has no valid observations.
    float32 reads match the source meteo precision; use dtype=np.float64 to
    retain all decimal precision in other ASC inputs. max_workers=1 is serial.
    Annual totals cover the supplied dates only; n_files records year coverage.
    bounds=(xmin, ymin, xmax, ymax), in the raster CRS, limits the actual read
    to all cells intersecting that extent. Bounds are clipped to the raster;
    bounds=None reads the full grid. No spatial resampling is performed.
    """
    file_names = list(file_names)
    if not file_names:
        raise ValueError('No .asc files provided.')
    if aggregate not in (None, 'year'):
        raise ValueError("aggregate must be None or 'year'.")
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError('max_workers must be a positive integer.')
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype must be float32 or float64.')

    records = []
    for file_name in file_names:
        path = Path(folder_path) / file_name
        match = re.search(r'(?:^|_)(\d{8})(?=_|$)', path.stem)
        if match is None:
            raise ValueError(f'No YYYYMMDD date in filename: {path.name}')
        records.append((datetime.strptime(match.group(1), '%Y%m%d'), path))
    records.sort(key=lambda item: item[0])
    dates, paths = zip(*records)
    if len(set(dates)) != len(dates):
        raise ValueError('Duplicate dates: provide exactly one raster per date.')

    # Read metadata once to allocate the output and establish the spatial grid.
    with rasterio.open(paths[0]) as src:
        shape = src.shape
        transform, crs = src.transform, src.crs
    source_grid = (shape, transform, crs)
    if transform.b != 0 or transform.d != 0:
        raise ValueError('Rotated grids cannot be represented by 1D x/y coordinates.')
    window = None
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError('bounds must contain four finite values: xmin, ymin, xmax, ymax.')
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError('bounds must have xmin < xmax and ymin < ymax.')
        requested = from_bounds(*bounds, transform=transform)
        edges = np.array(
            [
                requested.col_off,
                requested.row_off,
                requested.col_off + requested.width,
                requested.row_off + requested.height,
            ]
        )
        # Snap floating-point noise at cell edges before rounding outwards.
        nearest = np.rint(edges)
        edges = np.where(np.isclose(edges, nearest, rtol=0, atol=1e-8), nearest, edges)
        col_start, row_start = np.floor(edges[:2]).astype(int)
        col_stop, row_stop = np.ceil(edges[2:]).astype(int)
        col_start, row_start = max(0, col_start), max(0, row_start)
        col_stop, row_stop = min(shape[1], col_stop), min(shape[0], row_stop)
        if col_stop <= col_start or row_stop <= row_start:
            raise ValueError('bounds do not overlap the raster grid.')
        window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
        shape = (int(window.height), int(window.width))
        transform = window_transform(window, transform)
    x = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    y = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    years, year_indices, year_counts = np.unique([date.year for date in dates], return_inverse=True, return_counts=True)
    if aggregate == 'year':
        values = np.zeros((len(years), *shape), dtype=np.float64)
        has_data = np.zeros(values.shape, dtype=bool)
    else:
        # Allocate once rather than retaining a list and copying it with stack().
        values = np.empty((len(paths), *shape), dtype=dtype)

    workers = min(max_workers, os.cpu_count() or 1, len(paths))
    batch_size = 2 * workers
    read = partial(_read_asc_grid, dtype=dtype, window=window)
    next_report = 10
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Bound in-flight results: Executor.map on Python 3.13 otherwise queues
        # every file and may retain thousands of daily arrays.
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            for index, (data, grid) in enumerate(executor.map(read, batch), start=start):
                if data.shape != shape or grid != source_grid:
                    raise ValueError(f'Raster grid differs from the first file: {paths[index]}')
                if aggregate == 'year':
                    year_index = year_indices[index]
                    valid = ~np.isnan(data)
                    np.add(values[year_index], data, out=values[year_index], where=valid)
                    has_data[year_index] |= valid
                else:
                    values[index] = data
                files_read = index + 1
                percent = files_read * 100 // len(paths)
                if percent >= next_report:
                    elapsed = perf_counter() - started
                    print(
                        f'{name or "ASC"}: read {files_read}/{len(paths)} files '
                        f'({percent}%, {files_read / elapsed:.1f} files/s)',
                        flush=True,
                    )
                    next_report = (percent // 10 + 1) * 10

    attrs = {'source_format': 'Arc/Info ASCII Grid', 'transform': tuple(transform)[:6]}
    if crs is not None:
        attrs['crs'] = crs.to_string()
    if bounds is not None:
        attrs['requested_bounds'] = tuple(bounds)
    if aggregate == 'year':
        values[~has_data] = np.nan
        attrs['aggregation'] = 'Calendar-year sum of available dates; NoData skipped'
        return xr.DataArray(
            values,
            dims=('year', 'y', 'x'),
            coords={'year': years, 'y': y, 'x': x, 'n_files': ('year', year_counts)},
            name=name,
            attrs=attrs,
        )
    return xr.DataArray(
        values,
        dims=('time', 'y', 'x'),
        coords={'time': np.asarray(dates, dtype='datetime64[ns]'), 'y': y, 'x': x},
        name=name,
        attrs=attrs,
    )


# %% [markdown]
# # Options

# %%
M = Mdl_N('NBr150')

Pa_P = M.Pa.In / 'CAP/P/NBr119'
Pa_PET = M.Pa.In / 'CAP/PET/NBr119'
MAX_WORKERS = 16  # Benchmarked on these ASC files; lower this for slower disks.
BOUNDS = (M.Xmin, M.Ymin, M.Xmax, M.Ymax)

# %% [markdown]
# # Get file names

# %%
l_P_Fi = sorted(Pa_P.glob('P_*.asc'))
l_PET_Fi = sorted(Pa_PET.glob('PET_*.asc'))

# %% [markdown]
# # Annual sums
# Sum within the model bounds while reading, without retaining daily grids.
# n_files gives the number of available dates contributing to each year.

# %%
A_P_year = combine_asc_files(Pa_P, l_P_Fi, aggregate='year', name='P', max_workers=MAX_WORKERS, bounds=BOUNDS)

# %%
A_PET_year = combine_asc_files(Pa_PET, l_PET_Fi, aggregate='year', name='PET', max_workers=MAX_WORKERS, bounds=BOUNDS)

# %% [markdown]
# For daily arrays instead, use the following in place of the annual calls:
# ```python
# A_P = combine_asc_files(Pa_P, l_P_Fi, name='P', max_workers=MAX_WORKERS, bounds=BOUNDS)
# A_P_year = A_P.groupby('time.year').sum('time', skipna=True, min_count=1, dtype=np.float64)
# A_PET = combine_asc_files(Pa_PET, l_PET_Fi, name='PET', max_workers=MAX_WORKERS, bounds=BOUNDS)
# A_PET_year = A_PET.groupby('time.year').sum('time', skipna=True, min_count=1, dtype=np.float64)
# ```
