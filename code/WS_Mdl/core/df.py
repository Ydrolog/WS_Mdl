import numpy as np
import pandas as pd

from WS_Mdl.core.style import set_verbose, sprint
from WS_Mdl.io.ini import Mdl_Dmns_from_INI


def DF_info(DF: pd.DataFrame):
    """Prints basic info about a DataFrame."""
    print('dataframe info:')
    print(f'Shape: {DF.shape}')
    print(f'Data types:\n{DF.dtypes}')
    print('\nBasic statistics for numeric columns:')
    DF.describe()


def DF_memory(DF):
    """Returns human-readable memory usage of a DataFrame."""

    n = DF.memory_usage(deep=True).sum()
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.2f} {unit}'
        n /= 1024
    return f'{n:.2f} PB'


def DF_Col_value_counts_grouped(df, percentile_step=10):
    """Analyze DataFrame columns by non-null value counts, grouped into percentiles."""
    counts = {col: df[col].count() for col in df.columns}
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])

    results = []
    n_cols = len(sorted_counts)

    for i in range(0, 100, percentile_step):
        start_idx = int(i / 100 * n_cols)
        end_idx = int((i + percentile_step) / 100 * n_cols)
        if end_idx > n_cols:
            end_idx = n_cols
        if start_idx == end_idx and end_idx < n_cols:
            end_idx += 1

        if start_idx < n_cols:
            cols_in_range = sorted_counts[start_idx:end_idx]
            if cols_in_range:
                col_names = [c[0] for c in cols_in_range]
                col_counts = [c[1] for c in cols_in_range]
                results.append(
                    {
                        'Percentile_Range': f'{i}-{i + percentile_step}%',
                        'Min_Values': min(col_counts),
                        'Max_Values': max(col_counts),
                        'Num_Columns': len(col_names),
                        'Columns': col_names,
                    }
                )

    return pd.DataFrame(results)


def DF_Rd_Cols(DF, sig_figs=4):
    """
    Round all float columns in a pandas DataFrame to a specified number of significant figures.
    DF : pd.DataFrame : The DataFrame to round.
    sig_figs : int, optional : The number of significant figures to round to (default is 4).

    Returns: pd.DataFrame : A new DataFrame with rounded float columns.
    """
    DF_r = DF.copy()

    for col in DF_r.select_dtypes(include=['float', 'float64', 'float32']).columns:
        # Create a mask for non-zero and non-NaN values
        mask = (DF_r[col] != 0) & (DF_r[col].notna())
        if mask.any():
            vals = DF_r.loc[mask, col]
            # Calculate the number of decimal places needed for each value
            decimals = sig_figs - np.floor(np.log10(np.abs(vals))) - 1
            decimals = decimals.astype(int)

            # Apply rounding using multiplication method
            power_of_10 = 10.0**decimals
            DF_r.loc[mask, col] = np.around(vals * power_of_10) / power_of_10

    return DF_r


def GDF_clip_Mdl_Aa(GDF, Pa_INI):
    """Limits a GeoDataFrame to the model area defined in the INI file."""
    set_verbose(False)  # Suppress sprint from Mdl_Dmns_from_INI
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns_from_INI(Pa_INI)
    set_verbose(True)  # Re-enable sprint

    if all(col in GDF.columns for col in ['Xa', 'Ya', 'Xz', 'Yz']):
        GDF = GDF[
            (
                (GDF['Xa'].between(Xmin, Xmax, inclusive='both') | GDF['Xz'].between(Xmin, Xmax, inclusive='both'))
                & (GDF['Ya'].between(Ymin, Ymax, inclusive='both') | GDF['Yz'].between(Ymin, Ymax, inclusive='both'))
            )
        ]

    else:
        GDF = GDF[
            (
                (
                    GDF['Xstart'].between(Xmin, Xmax, inclusive='both')
                    | GDF['Xend'].between(Xmin, Xmax, inclusive='both')
                )
                & (
                    GDF['Ystart'].between(Ymin, Ymax, inclusive='both')
                    | GDF['Yend'].between(Ymin, Ymax, inclusive='both')
                )
            )
        ]

    sprint(
        f'🟢 - GeoDataFrame limited to model area from {Pa_INI}. Original rows: {len(GDF)}, Limited rows: {len(GDF)}.'
    )
    return GDF


def Calc_DF_XY(DF: pd.DataFrame, Xmin: float, Ymax: float, cellsize: float) -> pd.DataFrame:
    """
    Calculates X,Y coordinates for a DataFrame with row/column indices.
    Supports 3 naming convetions for now:
    - i, j
    - R, C
    - row, column
    """
    if ('i' in DF.columns) and ('j' in DF.columns):
        DF['X'] = Xmin + (DF['j'] - 0.5) * cellsize
        DF['Y'] = Ymax - (DF['i'] - 0.5) * cellsize
    elif ('R' in DF.columns) and ('C' in DF.columns):
        DF['X'] = Xmin + (DF['C'] - 0.5) * cellsize
        DF['Y'] = Ymax - (DF['R'] - 0.5) * cellsize
    elif ('row' in DF.columns) and ('column' in DF.columns):
        DF['X'] = Xmin + (DF['column'] - 0.5) * cellsize
        DF['Y'] = Ymax - (DF['row'] - 0.5) * cellsize
    else:
        sprint('🔴 - Cannot calculate coordinates: no suitable row/column indices found in PACKAGEDATA.')
        return
    return DF


def Calc_GDF_XY_start_end_from_Geom(DF: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates start and end X,Y coordinates from geometry column in a GeoDataFrame.
    Assumes there s a geometry column, of type shapely.geometry.
    """

    DF['Xa'] = DF['geometry'].apply(
        lambda x: x.geoms[0].coords[0][0]
    )  # Access X coorddinate of first point in first linestring
    DF['Ya'] = DF['geometry'].apply(lambda x: x.geoms[0].coords[0][1])
    DF['Xz'] = DF['geometry'].apply(lambda x: x.geoms[0].coords[-1][0])
    DF['Yz'] = DF['geometry'].apply(lambda x: x.geoms[0].coords[-1][1])

    return DF
