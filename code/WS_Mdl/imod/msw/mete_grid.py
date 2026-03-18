from pathlib import Path

import pandas as pd
from WS_Mdl.imod.prj import o_with_OBS


def to_DF(PRJ):
    """
    Reads the mete_grid.inp file specified in the PRJ and returns it as a DataFrame with an additional 'DT' column for datetime.

    Example usage:
    from WS_Mdl.imod.msw.mete_grid import to_DF
    from WS_Mdl.imod.msw.meteo import to_XA
    from WS_Mdl.imod.prj import r_with_OBS

    PRJ = r_with_OBS(M.Pa.PRJ)[0] # [0], cause [1] is the OBS
    DF_P = to_DF(PRJ)
    A_P = to_XA(DF_P, 'P', MdlN)
    """
    DF_meteo = pd.read_csv(PRJ['extra']['paths'][2][0], names=['day', 'year', 'P', 'PET'])
    DF_meteo['DT'] = pd.to_datetime(
        DF_meteo['year'].astype(int).astype(str) + '-' + (DF_meteo['day'].astype(int) + 1).astype(str), format='%Y-%j'
    )
    return DF_meteo


def add_missing_Cols(Pa, Pa_Out=None):
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


def Cvt_to_AbsPa(Pa_PRJ: Path | str, PRJ: dict = None):
    """
    Converts mete_grid.inp paths to absolute paths in the PRJ file.
    This is necessary because imod doesn't handle relative paths in mete_grid.inp correctly.
    - Pa_PRJ is necessary cause it is used in the path conversion.
    - PRJ is optional, if not provided, it will be loaded from Pa_PRJ.
    Returns Pa of mete_grid.inp with absolute paths.
    """
    Pa_PRJ = Path(Pa_PRJ)
    Dir_PRJ = Pa_PRJ.parent

    if not PRJ:  # If PRJ is not provided, load it from Pa_PRJ
        PRJ_, _ = o_with_OBS(Pa_PRJ)
        PRJ, _ = PRJ_[0], PRJ_[1]
        return None

    Pa_mete_grid = PRJ['extra']['paths'][2][0]  # 3rd file (index 2) (by default. immutable order)

    # Load mete_grid, edit and save it
    Pa_mete_grid_AbsPa = Pa_mete_grid.parent / 'temp' / 'mete_grid.inp'
    if not Pa_mete_grid_AbsPa.parent.exists():
        Pa_mete_grid_AbsPa.parent.mkdir(parents=True, exist_ok=True)

    DF = pd.read_csv(Pa_mete_grid, header=None, names=['N', 'Y', 'P', 'PET'])
    DF.P = DF.P.apply(lambda x: (Dir_PRJ / x).resolve())
    DF.PET = DF.PET.apply(lambda x: (Dir_PRJ / x).resolve())

    # Write CSV with proper format to avoid imod parsing issues with newlines
    # imod doesn't strip newlines from paths, so we need to format carefully
    corrected_lines = []
    for index, row in DF.iterrows():
        # Add quotes around paths like the original format
        line = f'{row["N"]},{row["Y"]},"{row["P"]}","{row["PET"]}"'
        corrected_lines.append(line)

    # Write without newlines in path columns
    with open(Pa_mete_grid_AbsPa, 'w') as f:
        for i, line in enumerate(corrected_lines):
            if i == len(corrected_lines) - 1:  # Last line - no newline
                f.write(line)
            else:
                f.write(line + '\n')

    print(f'Created corrected mete_grid.inp: {Pa_mete_grid_AbsPa}')

    return Pa_mete_grid_AbsPa
