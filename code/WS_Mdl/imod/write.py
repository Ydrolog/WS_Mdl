import pandas as pd
from filelock import FileLock as FL
from WS_Mdl.core.path import get_MdlN_Pa
from WS_Mdl.core.style import sprint

# region MF6 -----------------------------------------------------------------------------------------------------------------------


def DF_to_MF_block(DF, Min_width=4, indent='    ', Max_decimals=4):
    """
    Convert DataFrame to formatted MODFLOW input block.

    Creates a text block with consistent column widths, proper decimal formatting,
    and indentation for MODFLOW input files. The first column is commented with '#'.

    Parameters
    ----------
    DF : pandas.DataFrame
        DataFrame to format
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
    DF = DF.rename(columns={DF.columns[0]: '#' + DF.columns[0]})  # comment out header, so that MF6 doesn't read it.
    DF_str = DF.copy().astype(str)

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
    header_line = indent + ' '.join(col.rjust(widths[col]) for col in DF_str.columns)

    lines = [header_line]

    # Prepare data lines (right aligned)
    for _, row in DF_str.iterrows():
        line = indent + ' '.join(row[col].rjust(widths[col]) for col in DF_str.columns)
        lines.append(line)

    return '\n'.join(lines) + '\n'


def add_MVR_to_OPTIONS(Pa):
    """
    Opens a MODFLOW 6 input files (based on provided path), finds the OPTIONS block, and adds the MOVER option before the END OPTIONS line. Uses a file lock to ensure thread-safe file editing.

    Parameters
    ----------
    Pa : str
        Path to the MODFLOW 6 input file
    """
    lock = FL(f'{Pa}.lock')

    with lock:  # Acquire lock before editing file
        try:
            with open(Pa) as f:
                Lns = f.readlines()
            i = Lns.index('END OPTIONS\n')
            Lns[i] = '\tMOVER\nEND OPTIONS\n'
            with open(Pa, 'w') as f:
                f.writelines(Lns)
            sprint(f'🟢 - Added MOVER option to {Pa.name()}')
        except Exception as e:
            print(f'🔴 - Error adding MOVER option to {Pa.name()}: {e}')


def add_PKG_to_NAM(MdlN, str_PKG, iMOD5=False):
    """
    Adds a package (PKG) to the NAM file for the specified model (MdlN).
    """
    d_Pa = get_MdlN_Pa(MdlN, iMOD5=iMOD5)
    Pa_NAM = d_Pa['NAM_Mdl']

    lock = FL(Pa_NAM + '.lock')  # Create a file lock to prevent concurrent writes
    with lock, open(Pa_NAM, 'r+') as f:
        l_lines = f.readlines()
        l_lines[-1] = str_PKG
        f.seek(0)
        f.truncate()
        for i in l_lines:
            f.write(i)
        f.write('END PACKAGES')


def add_OBS_to_MF_In(str_OBS, PKG=None, MdlN=None, Pa=None, iMOD5=False):
    """
    Adds an OBS block to a MODFLOW 6 input file (to add to NAM, use utils.imod.py/add_OBS) (Pa). If Pa is not provided, it will be determined using MdlN and PKG.
    """

    if Pa is not None:
        Pa = Pa
    elif (MdlN is not None) and (PKG is not None):
        d_Pa = get_MdlN_Pa(MdlN, iMOD5=iMOD5)
        Pa = d_Pa['Sim_In'] / f'{MdlN}.{PKG}6'
    else:
        raise ValueError('Either Pa or both MdlN and PKG must be provided.')

    with open(Pa, 'r+') as f:
        l_Lns = f.readlines()
        try:
            i = next(i for i, ln in enumerate(l_Lns) if 'END OPTIONS' in ln.upper())
            l_1, l_2 = l_Lns[:i], l_Lns[i:]
            l_Lns = l_1 + [f'{str_OBS}\n'] + l_2
            f.seek(0)
            f.writelines(l_Lns)
            f.truncate()
            sprint(f'🟢 - Added OBS to {Pa}')
        except ValueError as e:
            print(f'🔴 - Failed:\n {e}')


# endregion ------------------------------------------------------------------------------------------------------------------------

# region MSW -----------------------------------------------------------------------------------------------------------------------


def mete_grid_add_missing_Cols(Pa, Pa_Out=None):
    """
    Add missing columns to the mete_grid.inp file if required:
    Ensures the file has 11 columns by adding default 'NoValue' entries for any
    """

    if Pa_Out is None:
        Pa_Out = Pa

    DF_mete_grid = pd.read_csv(Pa, header=None)

    if DF_mete_grid.shape[1] < 11:
        for col in range(DF_mete_grid.shape[1], 11):
            DF_mete_grid[col] = 'NoValue'  # Add missing columns with default value 'NoValue'
        DF_mete_grid.to_csv(Pa_Out, header=False, index=False, quoting=2)  # quoting=2 so that strings are quoted


# endregion ------------------------------------------------------------------------------------------------------------------------
