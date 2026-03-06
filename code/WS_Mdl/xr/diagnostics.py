import numpy as np
import pandas as pd
import xarray as xra


def _describe_da(da: xra.DataArray, da_name: str):
    """Helper function to describe a single DataArray."""
    print(f'--- Statistics for: {da_name} ---')

    da = da.load()

    if np.issubdtype(da.dtype, np.number):
        if da.count().item() > 0:
            stats = {
                'count': da.count().item(),
                'mean': da.mean().item(),
                'std': da.std().item(),
                'min': da.min().item(),
                '25%': da.quantile(0.25).item(),
                '50%': da.median().item(),
                '75%': da.quantile(0.75).item(),
                'max': da.max().item(),
            }
            value_stats = pd.Series(stats, name=da_name)
            print(value_stats)
        else:
            print('Array has no valid data (all NaNs or empty).')
    else:
        print(f'Variable is non-numeric (dtype: {da.dtype}).')
        unique_vals, counts = np.unique(da.values, return_counts=True)
        print('Unique values and their counts:')
        for val, count in zip(unique_vals, counts):
            print(f'  - {val}: {count}')

    print('\n--- Coordinate Summary ---')
    for coord_name in da.coords:
        coord = da.coords[coord_name]
        if coord.ndim == 1:
            summary = {'count': coord.size}

            c_min = coord.min().values
            c_max = coord.max().values

            if np.issubdtype(coord.dtype, np.datetime64):
                summary['min'] = pd.to_datetime(c_min).strftime('%Y-%m-%d')
                summary['max'] = pd.to_datetime(c_max).strftime('%Y-%m-%d')
            else:
                summary['min'] = c_min.item()
                summary['max'] = c_max.item()

            if np.issubdtype(coord.dtype, np.number) and coord.size > 1:
                diffs = np.diff(coord.values)
                if np.allclose(diffs, diffs[0]):
                    summary['step'] = diffs[0].item()

            print(f'- {coord_name} ({coord.dtype}):')
            print(pd.Series(summary).to_string())
            print()
    print('-' * 30)


def _compare_dataarrays(
    array1: xra.DataArray,
    array2: xra.DataArray,
    name1: str = 'Array 1',
    name2: str = 'Array 2',
    x_dim: str = 'x',
    y_dim: str = 'y',
    tolerance: float = 1e-10,
    title: str = None,
) -> dict:
    """Provides diagnostics for comparing two xarray DataArrays."""
    if title is None:
        title = f'=== Diagnostic Analysis: {name1} vs {name2} ==='

    print(title)
    print(f'{name1} shape: {array1.shape}')
    print(f'{name2} shape: {array2.shape}')
    print(f'{name1} dtype: {array1.dtype}')
    print(f'{name2} dtype: {array2.dtype}')

    results = {
        'shapes_identical': array1.shape == array2.shape,
        'dtypes_identical': array1.dtype == array2.dtype,
        'arrays_identical': array1.identical(array2),
        'arrays_equal': array1.equals(array2),
    }

    print(f'\nShapes identical: {results["shapes_identical"]}')

    coords_to_check = []
    if x_dim in array1.coords and x_dim in array2.coords:
        coords_to_check.append(x_dim)
    if y_dim in array1.coords and y_dim in array2.coords:
        coords_to_check.append(y_dim)

    for coord_name in coords_to_check:
        coord_identical = array1[coord_name].identical(array2[coord_name])
        results[f'{coord_name}_coords_identical'] = coord_identical
        print(f'{coord_name.upper()} coordinates identical: {coord_identical}')

        if not coord_identical:
            coord1, coord2 = array1.coords[coord_name], array2.coords[coord_name]
            print(f'  {coord_name.upper()} {name1} range: {coord1.min().values:.1f} to {coord1.max().values:.1f}')
            print(f'  {coord_name.upper()} {name2} range: {coord2.min().values:.1f} to {coord2.max().values:.1f}')

            if len(coord1) > 1:
                spacing1 = float(coord1.diff(coord_name)[0].values)
                print(f'  {coord_name.upper()} {name1} spacing: {spacing1:.1f}')
            if len(coord2) > 1:
                spacing2 = float(coord2.diff(coord_name)[0].values)
                print(f'  {coord_name.upper()} {name2} spacing: {spacing2:.1f}')

    print(f'Data values equal (equals): {results["arrays_equal"]}')

    try:
        if results['shapes_identical']:
            if x_dim in array2.coords and y_dim in array2.coords:
                array2_aligned = array2.interp({x_dim: array1[x_dim], y_dim: array1[y_dim]}, method='nearest')
            else:
                array2_aligned = array2

            diff = array1 - array2_aligned
            max_diff = abs(diff).max().values
            num_different = (abs(diff) > tolerance).sum().values

            results['max_absolute_difference'] = max_diff
            results['num_different_cells'] = num_different

            print(f'Maximum absolute difference (after alignment): {max_diff}')
            print(f'Number of different cells (tolerance={tolerance}): {num_different}')
        else:
            print('Cannot compare values directly due to different shapes')
            results['max_absolute_difference'] = None
            results['num_different_cells'] = None
    except Exception as e:
        print(f'Error comparing values: {e}')
        results['max_absolute_difference'] = None
        results['num_different_cells'] = None

    try:
        min1, max1 = array1.min().values, array1.max().values
        min2, max2 = array2.min().values, array2.max().values
        results['array1_range'] = (min1, max1)
        results['array2_range'] = (min2, max2)

        print(f'\n{name1} min/max: {min1}/{max1}')
        print(f'{name2} min/max: {min2}/{max2}')
    except Exception as e:
        print(f'Error calculating min/max: {e}')
        results['array1_range'] = None
        results['array2_range'] = None

    try:
        has_nan1 = array1.isnull().any().values
        has_nan2 = array2.isnull().any().values
        results['array1_has_nan'] = has_nan1
        results['array2_has_nan'] = has_nan2

        print(f'{name1} has NaN: {has_nan1}')
        print(f'{name2} has NaN: {has_nan2}')
    except Exception as e:
        print(f'Error checking for NaN values: {e}')
        results['array1_has_nan'] = None
        results['array2_has_nan'] = None

    return results


@xra.register_dataarray_accessor('ws')
class DataArrayAccessor:
    """Custom xarray DataArray accessor for WS diagnostics."""

    def __init__(self, xarray_obj: xra.DataArray):
        self._obj = xarray_obj

    def describe(self, name: str = None):
        """Describe the DataArray, including values and coordinate summary."""
        _describe_da(self._obj, name or self._obj.name or 'DataArray')

    def compare(
        self,
        other: xra.DataArray,
        name1: str = 'Array 1',
        name2: str = 'Array 2',
        x_dim: str = 'x',
        y_dim: str = 'y',
        tolerance: float = 1e-10,
        title: str = None,
    ) -> dict:
        """Compare this DataArray to another DataArray."""
        return _compare_dataarrays(
            self._obj,
            other,
            name1=name1,
            name2=name2,
            x_dim=x_dim,
            y_dim=y_dim,
            tolerance=tolerance,
            title=title,
        )


@xra.register_dataset_accessor('ws')
class DatasetAccessor:
    """Custom xarray Dataset accessor for WS diagnostics."""

    def __init__(self, xarray_obj: xra.Dataset):
        self._obj = xarray_obj

    def describe(self, name: str = None):
        """Describe each DataArray in the Dataset."""
        if name:
            print(f'--- Describing Dataset: {name} ---')
        for var_name, da in self._obj.data_vars.items():
            _describe_da(da, var_name)
