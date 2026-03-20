import numpy as np
import pandas as pd

from WS_Mdl.core.style import set_verbose, sprint
from WS_Mdl.imod.ini import Mdl_Dmns


@pd.api.extensions.register_dataframe_accessor('ws')
class DFAccessor:
    """Custom pandas DataFrame accessor providing WS_Mdl utility methods.

    Usage: import WS_Mdl.core.df # noqa: F401
    # registers the accessor. import WS_Mdl.core does that too, as it's defined in __init__.py # noqa: F401 blocks pyllance from removing the import upon saving (casue it's not used anywhere iself).
           df.ws.info()
           df.ws.memory()
           df.ws.round_Cols()
           ...
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def info(self):
        """Prints basic info about the DataFrame."""
        print('dataframe info:')
        print(f'\n - Shape: {self._df.shape}')
        print(f'\n - Data types:\n{self._df.dtypes}')
        print('\n - Basic statistics for numeric columns:')
        print(self._df.describe())

    def memory(self) -> str:
        """Returns human-readable memory usage of the DataFrame."""
        n = self._df.memory_usage(deep=True).sum()
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if n < 1024:
                return f'{n:.2f} {unit}'
            n /= 1024
        return f'{n:.2f} PB'

    def Col_value_counts_grouped(self, percentile_step: int = 10) -> pd.DataFrame:
        """Analyze DataFrame columns by non-null value counts, grouped into percentiles."""
        counts = {col: self._df[col].count() for col in self._df.columns}
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

    def round_Cols(self, sig_figs: int = 4) -> pd.DataFrame:
        """
        Round all float columns to a specified number of significant figures.
        sig_figs : int, optional : The number of significant figures to round to (default is 4).

        Returns: pd.DataFrame : A new DataFrame with rounded float columns.
        """
        df_r = self._df.copy()

        for col in df_r.select_dtypes(include=['float', 'float64', 'float32']).columns:
            # Create a mask for non-zero and non-NaN values
            mask = (df_r[col] != 0) & (df_r[col].notna())
            if mask.any():
                vals = df_r.loc[mask, col]
                # Calculate the number of decimal places needed for each value
                decimals = sig_figs - np.floor(np.log10(np.abs(vals))) - 1
                decimals = decimals.astype(int)

                # Apply rounding using multiplication method
                power_of_10 = 10.0**decimals
                df_r.loc[mask, col] = np.around(vals * power_of_10) / power_of_10

        return df_r

    def clip_Mdl_area(self, Pa_INI) -> pd.DataFrame:
        """Limits the DataFrame to the model area defined in the INI file."""
        set_verbose(False)  # Suppress sprint from Mdl_Dmns
        Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns(Pa_INI)
        set_verbose(True)  # Re-enable sprint

        n_original = len(self._df)

        if all(col in self._df.columns for col in ['Xa', 'Ya', 'Xz', 'Yz']):
            result = self._df[
                (
                    (
                        self._df['Xa'].between(Xmin, Xmax, inclusive='both')
                        | self._df['Xz'].between(Xmin, Xmax, inclusive='both')
                    )
                    & (
                        self._df['Ya'].between(Ymin, Ymax, inclusive='both')
                        | self._df['Yz'].between(Ymin, Ymax, inclusive='both')
                    )
                )
            ]
        else:
            result = self._df[
                (
                    (
                        self._df['Xstart'].between(Xmin, Xmax, inclusive='both')
                        | self._df['Xend'].between(Xmin, Xmax, inclusive='both')
                    )
                    & (
                        self._df['Ystart'].between(Ymin, Ymax, inclusive='both')
                        | self._df['Yend'].between(Ymin, Ymax, inclusive='both')
                    )
                )
            ]

        sprint(
            f'🟢 - DataFrame limited to model area from {Pa_INI}. Original rows: {n_original}, Limited rows: {len(result)}.'
        )
        return result

    def Calc_XY(self, Xmin: float, Ymax: float, cellsize: float) -> pd.DataFrame:
        """
        Calculates X,Y coordinates for row/column indices and returns a new DataFrame.
        Supports 3 naming conventions:
        - i, j
        - R, C
        - row, column
        """
        df = self._df.copy()
        if ('i' in df.columns) and ('j' in df.columns):
            df['X'] = Xmin + (df['j'] - 0.5) * cellsize
            df['Y'] = Ymax - (df['i'] - 0.5) * cellsize
        elif ('R' in df.columns) and ('C' in df.columns):
            df['X'] = Xmin + (df['C'] - 0.5) * cellsize
            df['Y'] = Ymax - (df['R'] - 0.5) * cellsize
        elif ('row' in df.columns) and ('column' in df.columns):
            df['X'] = Xmin + (df['column'] - 0.5) * cellsize
            df['Y'] = Ymax - (df['row'] - 0.5) * cellsize
        else:
            sprint('🔴 - Cannot calculate coordinates: no suitable row/column indices found in PACKAGEDATA.')
            return self._df
        return df

    def Calc_XY_start_end_from_Geom(self) -> pd.DataFrame:
        """
        Calculates start and end X,Y coordinates from geometry column.
        Assumes there is a geometry column of type shapely.geometry.
        """
        df = self._df.copy()
        df['Xa'] = df['geometry'].apply(
            lambda x: x.geoms[0].coords[0][0]
        )  # Access X coordinate of first point in first linestring
        df['Ya'] = df['geometry'].apply(lambda x: x.geoms[0].coords[0][1])
        df['Xz'] = df['geometry'].apply(lambda x: x.geoms[0].coords[-1][0])
        df['Yz'] = df['geometry'].apply(lambda x: x.geoms[0].coords[-1][1])
        return df

        return self

    def to_MF_block(self, DF=None, Min_width=4, indent: int = 2, Max_decimals=4):
        """
        Convert DataFrame to formatted MODFLOW input block.

        Creates a text block with consistent column widths, proper decimal formatting,
        and indentation for MODFLOW input files. The first column is commented with '#'.

        Parameters
        ----------
        DF : pandas.DataFrame, optional
            DataFrame to format. If omitted, uses the accessor DataFrame.
        Min_width : int, default=4
            Minimum width for each column
        indent : str, default='    '
            String for line indentation
        Max_decimals : int, default=4
            Maximum decimal places for float values

        Returns
        -------
        str
            Formatted text block with right-aligned columns and consistent formatting
        """
        df_src = self._df if DF is None else DF
        if not isinstance(df_src, pd.DataFrame):
            raise TypeError('DF must be a pandas DataFrame when provided.')
        if len(df_src.columns) == 0:
            raise ValueError('Cannot format an empty DataFrame with no columns.')

        # Comment out the first header, so that MF6 does not read it.
        DF_fmt = df_src.rename(columns={df_src.columns[0]: '#' + df_src.columns[0]})
        DF_str = DF_fmt.copy().astype(str)

        # Detect columns with floats or decimals
        decimal_cols = []
        for col in DF_str.columns:
            # Try converting to float, if success and decimals exist, mark column
            try:
                floats = DF_str[col].astype(float)
                # Check if any value has decimals
                if any(floats % 1 != 0):
                    decimal_cols.append(col)
            except Exception:
                continue

        # Determine max decimals per decimal column
        max_decimals = {}
        for col in decimal_cols:
            floats = DF_str[col].astype(float)
            # Count decimals per value
            decimals_count = [len(str(f).split('.')[-1]) if '.' in str(f) else 0 for f in DF_str[col]]
            max_decimals[col] = min(max(decimals_count), Max_decimals)

        # Format decimal columns with fixed decimals, pad zeros
        for col in decimal_cols:
            decimals = max_decimals[col]
            DF_str[col] = DF_str[col].astype(float).map(lambda x: f'{x:.{decimals}f}')

        # Convert all columns to strings after formatting decimals
        DF_str = DF_str.astype(str)

        # Compute width per column: max(header length, max value length, Min_width)
        widths = {}
        for col in DF_str.columns:
            max_val_len = DF_str[col].map(len).max()
            widths[col] = max(len(col), max_val_len, Min_width)

        # Prepare header line (right aligned)
        header_line = indent * '  ' + ' '.join(col.rjust(widths[col]) for col in DF_str.columns)

        lines = [header_line]

        # Prepare data lines (right aligned)
        for _, row in DF_str.iterrows():
            line = indent * '  ' + ' '.join(row[col].rjust(widths[col]) for col in DF_str.columns)
            lines.append(line)

        return '\n'.join(lines) + '\n'
