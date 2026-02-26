import pandas as pd


def DF_info(DF: pd.DataFrame):
    """Prints basic info about a DataFrame."""
    print('Lines dataframe info:')
    print(f'Shape: {DF.shape}')
    print(f'Data types:\n{DF.dtypes}')
    print('\nBasic statistics for numeric columns:')
    DF.describe()
