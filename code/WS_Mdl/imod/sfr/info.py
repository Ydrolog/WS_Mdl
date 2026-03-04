import pandas as pd
from WS_Mdl.core.mdln import Calc_DF_XY, MdlN
from WS_Mdl.core.style import sprint
from WS_Mdl.io.text import r_Txt_Lns


def SFR_PkgD_to_DF(MdlN_1: str, Pa_SFR: str = None, Calc_Cond=True, iMOD5: bool = None) -> pd.DataFrame:
    """
    Reads SFR6 PACKAGE DATA block from a .SFR6 file, from MdlN folder, and returns it as a pandas DataFrame.
    Pa_SFR: Path to the SFR6 file. If None, it will be determined using MdlN_Pa().
    iMOD5: Boolean indicating whether to use the imod5 folder structure. If None, it will be determined automatically.
    """
    M = MdlN(MdlN_1)

    if Pa_SFR is None:
        Pa_SFR = M.Pa.SFR

    l_Lns = r_Txt_Lns(Pa_SFR)

    PkgDt_start = next(i for i, ln in enumerate(l_Lns) if 'BEGIN PACKAGEDATA' in ln.upper()) + 2
    PkgDt_end = next(i for i, ln in enumerate(l_Lns) if 'END PACKAGEDATA' in ln.upper())
    PkgDt_Cols = l_Lns[PkgDt_start - 1].replace('#', '').strip().split()
    PkgDt_data = [ln.split() for ln in l_Lns[PkgDt_start:PkgDt_end] if ln.strip() and not ln.strip().startswith('#')]

    for row in (
        PkgDt_data
    ):  # Reaches with cellid NONE (unconnected) are problematic cause other cellids have 3 values (L, R, C)
        if row.count('NONE') == 1 and row[1] == 'NONE':
            row[1:2] = ['NONE', 'NONE', 'NONE']

    DF = pd.DataFrame(PkgDt_data, columns=PkgDt_Cols)

    DF = DF.replace(['NONE', '', 'NaN', 'nan'], pd.NA)  # 1. normalize NA-like tokens and strip spaces
    DF = DF.apply(lambda s: s.str.strip() if s.dtype == 'object' else s)

    l_Num_Cols = [c for c in DF.columns if c != 'aux']  # 2. choose numeric columns and coerce
    DF[l_Num_Cols] = DF[l_Num_Cols].apply(pd.to_numeric)

    DF = DF.convert_dtypes()  # 3. optional: get nullable ints/floats

    if ('X' not in PkgDt_Cols) or ('Y' not in PkgDt_Cols):
        sprint('🟡 - Coordinates (X, Y columns) not found in PACKAGEDATA. Calculating coordinates from INI file info.')
        Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = M.Dmns(M.Pa.INI)
        DF = Calc_DF_XY(DF, Xmin, Ymax, cellsize)

    if Calc_Cond:
        if ('rlen' in DF.columns) and ('rwid' in DF.columns) and ('rthick' in DF.columns) and ('rcond' in DF.columns):
            DF['Cond'] = DF['rlen'] * DF['rwid'] * DF['rthick'] / DF['rcond']
        else:
            DF['Cond'] = DF.iloc[:, 4] * DF.iloc[:, 5] * DF.iloc[:, 9] / DF.iloc[:, 8]
        DF.insert(4, 'Cond', DF.pop('Cond'))  # Move Cond to column 4

    return DF


def SFR_ConnD_to_DF(MdlN_1: str, Pa_SFR: str = None, iMOD5: bool = None) -> pd.DataFrame:
    """
    Reads SFR6 connection data from a .SFR6 file and returns it as a pandas DataFrame.
    """
    M = MdlN(MdlN_1)

    if Pa_SFR is None:
        Pa_SFR = M.Pa.SFR

    l_Lns = r_Txt_Lns(Pa_SFR)

    Conn_start = next(i for i, ln in enumerate(l_Lns) if 'BEGIN CONNECTIONDATA' in ln.upper()) + 1
    Conn_end = next(i for i, ln in enumerate(l_Lns) if 'END CONNECTIONDATA' in ln.upper())
    Conn_data = [
        (int(parts[0]), [int(x) for x in parts[1:]])
        for ln in l_Lns[Conn_start:Conn_end]
        if (parts := ln.strip().split())
    ]

    DF_Conn = pd.DataFrame(Conn_data, columns=['reach_N', 'connections'])
    DF_Conn['downstream'] = DF_Conn['connections'].apply(lambda l_Conns: next((-x for x in l_Conns if x < 0), None))
    DF_Conn['downstream'] = DF_Conn['downstream'].astype('Int64')

    return DF_Conn


def reach_to_cell_id(reach: int, GDF_SFR: pd.DataFrame, MdlN=None, reach_Col='rno', L_C='k', r_C='i', C_C='j'):
    """Returns the cell_id (L, R, C) tuple for a given reach number using the provided SFR GeoDataFrame."""
    if (GDF_SFR is None) and (MdlN is None):
        raise ValueError('Either GDF_SFR or MdlN must be provided.')
    if MdlN:
        GDF_SFR = SFR_PkgD_to_DF(MdlN)

    if reach not in GDF_SFR[reach_Col].values:
        raise ValueError(f'Reach number {reach} not found in the model.')

    L = GDF_SFR.loc[GDF_SFR[reach_Col] == reach, L_C].values[0]
    R = GDF_SFR.loc[GDF_SFR[reach_Col] == reach, r_C].values[0]
    C = GDF_SFR.loc[GDF_SFR[reach_Col] == reach, C_C].values[0]

    return (L, R, C)


def reach_to_XY(reach: int, GDF_SFR: pd.DataFrame, MdlN=None, reach_Col='rno', X_C='X', Y_C='Y'):
    """Returns the X, Y coordinates for a given reach number using the provided SFR GeoDataFrame."""
    if (GDF_SFR is None) and (MdlN is None):
        raise ValueError('Either GDF_SFR or MdlN must be provided.')
    if MdlN:
        GDF_SFR = SFR_PkgD_to_DF(MdlN)

    if reach not in GDF_SFR[reach_Col].values:
        raise ValueError(f'Reach number {reach} not found in the model.')

    X = GDF_SFR.loc[GDF_SFR[reach_Col] == reach, X_C].values[0]
    Y = GDF_SFR.loc[GDF_SFR[reach_Col] == reach, Y_C].values[0]

    return (X, Y)


def get_SFR_OBS_Out_Pas(MdlN, filetype='.csv'):
    """
    Gets the SFR OBS output paths from the model simulation input folder.
    Returns a list of paths if multiple found, or a single path if only one found.
    666 need to add argument to select specific OBS file if arg is not None.
    """
    M = MdlN(MdlN)
    l_Pa = [
        M.Pa.Sim_In / i
        for i in M.Pa.Sim_In.iterdir()
        if 'SFR' in i.name.upper() and 'OBS' in i.name.upper() and i.name.lower().endswith(filetype)
    ]
    if len(l_Pa) == 1:
        l_Pa = l_Pa[0]
    return l_Pa
