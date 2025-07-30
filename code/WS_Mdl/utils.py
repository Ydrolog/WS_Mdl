# ***** Utility functions to facilitate more robust modelling. *****
import os
import re
import shutil as sh
import subprocess as sp
import warnings
from datetime import datetime as DT
from io import StringIO
from multiprocessing import Pool, cpu_count
from os import listdir as LD
from os.path import basename as PBN
from os.path import join as PJ
from pathlib import Path

import pandas as pd
from colored import attr, fg
from filelock import FileLock as FL

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.worksheet._read_only')

# --------------------------------------------------------------------------------
Pre_Sign = f'{fg(52)}{"*" * 80}{attr("reset")}\n'
Sign = f'{fg(52)}\nend_of_transmission\n{"*" * 80}{attr("reset")}\n'
bold = '\033[1m'
bold_off = '\033[0m'
custom_characters = ['🔴', '🟡', '🟢']

VERBOSE = True  # Use set_verbose to change this to true and get more information printed to the console.

Pa_WS = 'C:/OD/WS_Mdl'
Pa_RunLog = PJ(Pa_WS, 'Mng/WS_RunLog.xlsx')
Pa_log = PJ(Pa_WS, 'Mng/log.csv')
## Can make get paths function that will provide the general directories, like Pa_WS, Pa_Mdl. Those can be derived from a folder structure.


# VERBOSE ------------------------------------------------------------------------
def vprint(*args, **kwargs):
    """Prints only if VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


def set_verbose(v: bool):
    """Sets the VERBOSE variable to True or False."""
    global VERBOSE
    VERBOSE = v


# --------------------------------------------------------------------------------


# Get MdlN info ------------------------------------------------------------------
def MdlN_Se_from_RunLog(
    MdlN,
):  # Can be made faster. May need to make excel export the RunLog as a csv, so that I can use pd.read_csv instead of pd.read_excel.
    """Returns RunLog line that corresponds to MdlN as a S."""

    DF = pd.read_excel(PJ(Pa_WS, 'Mng/WS_RunLog.xlsx'), sheet_name='RunLog')
    Se_match = DF.loc[DF['MdlN'].str.lower() == MdlN.lower()]  # Match MdlN, case insensitive.
    if Se_match.empty:
        raise ValueError(
            f'MdlN {MdlN} not found in RunLog. {fg("indian_red_1c")}Check the spelling and try again.{attr("reset")}'
        )
    S = Se_match.squeeze()
    return S


def paths_from_MdlN_Se(S, MdlN):
    """Takes in S, returns relevant paths."""
    Mdl, SimN_B = S[['model alias', 'B SimN']]
    MdlN_B = Mdl + str(SimN_B)

    ## Non paths + General paths
    d_Pa = {}
    d_Pa['Mdl'] = Mdl
    d_Pa['MdlN'] = MdlN
    d_Pa['Pa_Mdl'] = PJ(Pa_WS, f'models/{Mdl}')
    d_Pa['Smk_temp'] = PJ(d_Pa['Pa_Mdl'], 'code/snakemake/temp')
    d_Pa['PoP'] = PJ(d_Pa['Pa_Mdl'], 'PoP')

    ## S Sim paths (grouped based on: pre-run, run, post-run)
    d_Pa['INI'] = PJ(d_Pa['Pa_Mdl'], f'code/Mdl_Prep/Mdl_Prep_{MdlN}.ini')
    d_Pa['BAT'] = PJ(d_Pa['Pa_Mdl'], f'code/Mdl_Prep/Mdl_Prep_{MdlN}.bat')
    d_Pa['PRJ'] = PJ(d_Pa['Pa_Mdl'], f'In/PRJ/{MdlN}.prj')
    d_Pa['Smk'] = PJ(d_Pa['Pa_Mdl'], f'code/snakemake/{MdlN}.smk')

    d_Pa['Pa_MdlN'] = PJ(d_Pa['Pa_Mdl'], f'Sim/{MdlN}')
    d_Pa['LST_Sim'] = PJ(d_Pa['Pa_MdlN'], 'mfsim.lst')  # Sim LST file
    d_Pa['LST_Mdl'] = PJ(d_Pa['Pa_MdlN'], f'GWF_1/{MdlN}.lst')  # Mdl LST file
    d_Pa['NAM_Sim'] = PJ(d_Pa['Pa_MdlN'], 'MFSIM.NAM')  # Sim LST file
    d_Pa['NAM_Mdl'] = PJ(d_Pa['Pa_MdlN'], f'GWF_1/{MdlN}.NAM')  # Mdl LST file

    d_Pa['Out_HD'] = PJ(d_Pa['Pa_MdlN'], 'GWF_1/MODELOUTPUT/HEAD/HEAD')
    d_Pa['PoP_Out_MdlN'] = PJ(d_Pa['PoP'], 'Out', MdlN)
    d_Pa['MM'] = PJ(d_Pa['PoP_Out_MdlN'], f'MM-{MdlN}.qgz')

    ## B Sim paths
    for k in list(d_Pa.keys()):
        if f'{k}_B' not in d_Pa:
            d_Pa[f'{k}_B'] = d_Pa[k].replace(MdlN, MdlN_B)

    return d_Pa


def get_MdlN_paths(MdlN: str, verbose=False):
    """
    Returns a dictionary of useful objects (mainly paths, but also Mdl, MdlN) for a given MdlN. Those need to then be passed to arguments, e.g.:
    d_Pa = get_MdlN_paths(MdlN)
    Pa_INI_B = d_Pa['Pa_INI_B'].
    """
    d_Pa = paths_from_MdlN_Se(MdlN_Se_from_RunLog((MdlN)), MdlN)
    vprint(f'🟢 - {MdlN} paths extracted from RunLog and returned as dictionary with keys:\n{", ".join(d_Pa.keys())}')
    return d_Pa


def get_MdlN_Pa(MdlN: str, MdlN_B=None, verbose=False):
    """
    #Improved get_MdlN_paths.#
    - Doesn't read RunLog, unless B is set to True. Thus it's much faster.
    - Returns a dictionary of useful objects (mainly paths, but also Mdl, MdlN) for a given MdlN. Those need to then be passed to arguments, e.g.
        d_Pa = get_MdlN_Pa(MdlN)
        Pa_INI = d_Pa['Pa_INI'].
    """

    if MdlN_B:
        d_Pa = paths_from_MdlN_Se(MdlN_Se_from_RunLog((MdlN)), MdlN)
    else:
        ## Non paths + General paths
        Mdl = get_Mdl(MdlN)  # Get model alias from MdlN

        d_Pa = {}
        d_Pa['Mdl'] = Mdl
        d_Pa['MdlN'] = MdlN
        d_Pa['Pa_Mdl'] = PJ(Pa_WS, f'models/{Mdl}')
        d_Pa['Smk_temp'] = PJ(d_Pa['Pa_Mdl'], 'code/snakemake/temp')
        d_Pa['PoP'] = PJ(d_Pa['Pa_Mdl'], 'PoP')

        ## S Sim paths (grouped based on: pre-run, run, post-run)
        d_Pa['INI'] = PJ(d_Pa['Pa_Mdl'], f'code/Mdl_Prep/Mdl_Prep_{MdlN}.ini')
        d_Pa['BAT'] = PJ(d_Pa['Pa_Mdl'], f'code/Mdl_Prep/Mdl_Prep_{MdlN}.bat')
        d_Pa['PRJ'] = PJ(d_Pa['Pa_Mdl'], f'In/PRJ/{MdlN}.prj')
        d_Pa['Smk'] = PJ(d_Pa['Pa_Mdl'], f'code/snakemake/{MdlN}.smk')

        d_Pa['Pa_MdlN'] = PJ(d_Pa['Pa_Mdl'], f'Sim/{MdlN}')
        d_Pa['LST_Sim'] = PJ(d_Pa['Pa_MdlN'], 'mfsim.lst')  # Sim LST file
        d_Pa['LST_Mdl'] = PJ(d_Pa['Pa_MdlN'], f'GWF_1/{MdlN}.lst')  # Mdl LST file
        d_Pa['NAM_Sim'] = PJ(d_Pa['Pa_MdlN'], 'MFSIM.NAM')  # Sim LST file
        d_Pa['NAM_Mdl'] = PJ(d_Pa['Pa_MdlN'], f'GWF_1/{MdlN}.NAM')  # Mdl LST file

        d_Pa['Out_HD'] = PJ(d_Pa['Pa_MdlN'], 'GWF_1/MODELOUTPUT/HEAD/HEAD')
        d_Pa['PoP_Out_MdlN'] = PJ(d_Pa['PoP'], 'Out', MdlN)
        d_Pa['MM'] = PJ(d_Pa['PoP_Out_MdlN'], f'MM-{MdlN}.qgz')

        if MdlN_B:  ## B Sim paths
            for k in list(d_Pa.keys()):
                if f'{k}_B' not in d_Pa:
                    d_Pa[f'{k}_B'] = d_Pa[k].replace(MdlN, MdlN_B)
    return d_Pa


def get_Mdl(MdlN: str):
    """Returns model alias for a given MdlN."""
    return ''.join([i for i in MdlN if i.isalpha()])


def get_last_MdlN():
    DF = pd.read_csv(Pa_log)
    DF.loc[:-2, 'Sim end DT'] = DF.loc[:-2, 'Sim end DT'].apply(pd.to_datetime, dayfirst=True)
    DF['Sim end DT'] = pd.to_datetime(DF['Sim end DT'], format='mixed', dayfirst=True)
    return DF.sort_values('Sim end DT', ascending=False).iloc[0]['MdlN']


# --------------------------------------------------------------------------------


# READ FILES ---------------------------------------------------------------------
# 666 to be iproved later by replacing paths with MdlN. I'll have to make get_MdlN_paths_noB, where RunLog won't be read. Path of one MdlN will be calculated off of standard folder structure.
def read_RunLog():
    return pd.read_excel(Pa_RunLog, sheet_name='RunLog').dropna(subset='runN')  # Read RunLog


def read_IPF_Spa(Pa_IPF):
    """Reads IPF file without temporal component - i.e. no linked TS text files. Returns a DF created from just the IPF file."""
    with open(Pa_IPF, 'r') as f:
        l_Ln = f.readlines()

    N_C = int(l_Ln[1].strip())  # Number of columns
    l_C_Nm = [l_Ln[i + 2].split('\n')[0] for i in range(N_C)]  # Extract column names
    DF_IPF = pd.read_csv(Pa_IPF, skiprows=2 + N_C + 1, names=l_C_Nm)

    vprint(
        f'🟢 - IPF file {Pa_IPF} read successfully. DataFrame created with {len(DF_IPF)} rows and {len(DF_IPF.columns)} columns.'
    )
    return DF_IPF


def INI_to_d(Pa_INI: str) -> dict:
    """Reads INI file (used for preparing the model) and converts it to a dictionary. Keys are converted to upper-case.
    Common use:
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize )
    print(f'The model area has {N_R} rows and {N_C} columns.')
    """
    d_INI = {}
    with open(Pa_INI, 'r', encoding='utf-8') as file:
        for Ln in file:
            Ln = Ln.strip()
            if Ln and not Ln.startswith('#'):  # Ignore empty lines and comments
                k, v = Ln.split('=', 1)  # Split at the first '='
                d_INI[k.strip().upper()] = v.strip()  # Remove extra spaces

    vprint(f'🟢 - INI file {Pa_INI} read successfully. Dictionary created with {len(d_INI)} keys.')
    return d_INI


def Mdl_Dmns_from_INI(
    Pa_INI,
):  # 666 Can be improved. It should take a MdlN instead of a path. Makes things easier.
    """
    Returns model dimension parameters. Common use:
    import WS_Mdl.utils as U
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = U.Mdl_Dmns_from_INI(path)
    """
    d_INI = INI_to_d(Pa_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    N_R, N_C = int(-(Ymin - Ymax) / cellsize), int((Xmax - Xmin) / cellsize)

    vprint(f'🟢 - model dimensions extracted from {Pa_INI}.')
    return Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C


def HD_Out_IDF_to_DF(
    path, add_extra_cols: bool = True
):  # 666 can make it save DF (e.g. to CSV) if a 2nd path is provided. Unecessary for now.
    """
    Reads all .IDF files in `path` into a DataFrame with columns:
      - path, file, type, year, month, day, L
    If add_extra_cols=True, also adds:
      - season (Winter/Spring/Summer/Autumn)
      - season_year (roll Winter Dec→Feb into next calendar year)
      - quarter (Q1-Q4)
      - Hy_year (hydrological year: Oct-Sep → Oct-Dec roll into next year)

      Parameters are extracted from filnames, based on a standard format. Hence, don't use this for other groups of IDF files, unless you're sure they follow the same format."""  # 666 can be generalized later, to work on all sorts of IDF files.

    Se_Fi_path = pd.Series([PJ(path, i) for i in LD(path) if i.lower().endswith('.idf')])
    DF = pd.DataFrame({'path': Se_Fi_path, 'file': Se_Fi_path.apply(lambda x: PBN(x))})
    DF[['type', 'year', 'month', 'day', 'L']] = (
        DF['file']
        .str.extract(r'^(?P<type>[A-Z]+)_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\d{6}_L(?P<L>\d+)\.IDF$')
        .astype({'year': int, 'month': int, 'day': int, 'L': int})
    )

    if add_extra_cols:
        # 1) season & season_year
        month2season = {
            12: 'Winter',
            1: 'Winter',
            2: 'Winter',
            3: 'Spring',
            4: 'Spring',
            5: 'Spring',
            6: 'Summer',
            7: 'Summer',
            8: 'Summer',
            9: 'Autumn',
            10: 'Autumn',
            11: 'Autumn',
        }

        DF['season'] = DF['month'].map(month2season)
        DF['season_year'] = DF.apply(
            lambda r: r.year + 1 if r.month == 12 else r.year, axis=1
        )  # roll December into next year's winter

        # 2) quarter (calendar)
        DF['quarter'] = DF['month'].apply(lambda m: f'Q{((m - 1) // 3) + 1}')

        # 3) GHG “water” year (Apr–Mar) months 4–12 → water_year = year+1; months 1–3 → water_year = year
        DF['GW_year'] = DF.apply(lambda r: r.year if r.month >= 4 else r.year - 1, axis=1)

    # DF.to_csv(PJ(path, 'contents.csv'), index=False)

    return DF


def MF6_block_to_DF(
    Pa: str | Path,
    block: str,
    *,
    comment_chars: tuple[str, ...] = ('#', '!', '//'),
    has_header: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read the lines between ``BEGIN <block>`` and ``END <block>`` in a MODFLOW-6 file
    into a pandas DataFrame.

    A single comment line that **begins with ``#`` directly in front of the first data
    row** (i.e. before any non-comment, non-blank line is seen) is interpreted as the
    column names for the resulting DF, regardless of *has_header*.

    Parameters
    ----------
    Pa : str or Path
        Path to the MF6 file.
    block : str
        Block name exactly as it appears after BEGIN / END (case-insensitive).
    comment_chars : tuple[str, ...], optional
        One-character prefixes that mark a line as a comment and should be skipped.
    has_header : bool, optional
        If True *and* no leading ``#`` header is found, the first non-comment line in
        the block is treated as column names. If False, columns will be numbered
        ``col_0, col_1, …``.
    **read_csv_kwargs
        Extra arguments forwarded to :pyfunc:`pandas.read_csv` (dtype, sep, na_values,
        etc.).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the block's tabular data.
    """
    Pa = Path(Pa)
    begin_pat = re.compile(rf'^\s*BEGIN\s+{re.escape(block)}\s*$', re.IGNORECASE)
    end_pat = re.compile(rf'^\s*END\s+{re.escape(block)}\s*$', re.IGNORECASE)

    capture = False
    buffer: list[str] = []
    header_line: str | None = None

    with Pa.open('r', encoding='utf-8') as f:
        for line in f:
            if not capture and begin_pat.match(line):
                capture = True
                continue  # don’t include the BEGIN line itself
            if capture:
                if end_pat.match(line):
                    break  # reached END <block>

                stripped = line.lstrip()

                # Detect and store a single leading '#' comment as header names
                if any(stripped.startswith(c) for c in comment_chars):
                    if (
                        stripped.startswith('#')
                        and header_line is None
                        and not buffer  # only if it's immediately before data
                    ):
                        header_line = stripped[1:].strip()
                    continue  # skip all comment lines

                if not stripped:
                    continue  # skip blank lines

                buffer.append(line)

    if not buffer:
        raise ValueError(f"Block '{block}' not found or contained no data in {Pa}")

    text = ''.join(buffer)

    # Determine how pandas will treat headers inside *text*
    pandas_header = None if header_line is not None else (0 if has_header else None)

    DF = pd.read_csv(
        StringIO(text),
        delim_whitespace=True,  # MF6 tables are whitespace-delimited
        header=pandas_header,
        comment=None,  # comments were already handled
        **read_csv_kwargs,
    )

    # Apply column names logic
    if header_line is not None:
        DF.columns = re.split(r'\s+', header_line.strip())
    elif not has_header:
        DF.columns = [f'col_{i}' for i in range(DF.shape[1])]

    return DF


# --------------------------------------------------------------------------------


# Open files ---------------------------------------------------------------------
def open_(key, *l_MdlN, Pa=r'C:\Program Files\Notepad++\notepad++.exe'):
    """Opens files at default locations, as specified by get_MdlN_Pa()."""
    if key not in get_MdlN_Pa('NBr1').keys():
        raise ValueError(f'\nInvalid key: {key}.\nValid keys are: {", ".join(get_MdlN_Pa("NBr1").keys())}')
        return

    vprint(f'{"-" * 100}\nOpening {key} file(s) for specified run(s) with the default program.\n')
    vprint(
        f"It's assumed that Notepad++ is installed in: {Pa}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument (Pa) to this function.\n"
    )

    l_Pa = [get_MdlN_Pa(MdlN)[key] for MdlN in l_MdlN]

    for f in l_Pa:
        sp.Popen([Pa] + [f])
        vprint(f'🟢 - {f}')


def Sim_Cfg(*l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    vprint(
        f"{'-' * 100}\nOpening all configuration files for specified runs with the default program.\nIt's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['Smk', 'BAT', 'INI', 'PRJ']
    l_paths = [get_MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]
    sp.Popen([Pa_NP] + l_files)
    for f in l_files:
        vprint(f'🟢 - {f}')


def open_LSTs(*l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    vprint(f'{"-" * 100}\nOpening LST files (Mdl+Sim) for specified runs with the default program.\n')
    vprint(
        f"It's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['LST_Sim', 'LST_Mdl']
    l_paths = [get_MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]

    for f in l_files:
        sp.Popen([Pa_NP] + [f])
        vprint(f'🟢 - {f}')


def open_NAMs(*l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    vprint(f'{"-" * 100}\nOpening NAM files (Mdl+Sim) for specified runs with the default program.\n')
    vprint(
        f"It's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['NAM_Sim', 'NAM_Mdl']
    l_paths = [get_MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]

    for f in l_files:
        sp.Popen([Pa_NP] + [f])
        vprint(f'🟢 - {f}')


def open_LST(
    *l_MdlN, Pa_NP=r'C:\Program Files\Notepad++\notepad++.exe'
):  # 666 To be deprecated later, as open_ does the same thing, but is more versatile.
    vprint(f'{"-" * 100}\nOpening LST files (Mdl+Sim) for specified runs with the default program.\n')
    vprint(
        f"It's assumed that Notepad++ is installed in: {Pa_NP}.\nIf that's not True, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n"
    )

    l_keys = ['LST_Mdl']
    l_paths = [get_MdlN_Pa(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]

    for f in l_files:
        sp.Popen([Pa_NP] + [f])
        vprint(f'🟢 - {f}')


# --------------------------------------------------------------------------------


# Formatting ---------------------------------------------------------------------
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


# --------------------------------------------------------------------------------


# Sim Prep + Run -----------------------------------------------------------------
def S_from_B(MdlN: str):
    """Copies files that contain Sim options from the B Sim, renames them for the S Sim, and opens them in the default file editor. Assumes default WS_Mdl folder structure (as described in READ_ME.MD)."""
    vprint(Pre_Sign)
    d_Pa = get_MdlN_paths(MdlN)  # Get default directories
    MdlN_B, Pa_INI_B, Pa_INI, Pa_BAT_B, Pa_BAT, Pa_Smk, Pa_Smk_B, Pa_PRJ_B, Pa_PRJ = (
        d_Pa[k] for k in ['MdlN_B', 'INI_B', 'INI', 'BAT_B', 'BAT', 'Smk', 'Smk_B', 'PRJ_B', 'PRJ']
    )  # and pass them to objects that will be used in the function

    # Copy .INI, .bat, .prj and make default (those apply to every Sim) modifications
    for Pa_B, Pa_S in zip([Pa_Smk_B, Pa_BAT_B, Pa_INI_B], [Pa_Smk, Pa_BAT, Pa_INI]):
        try:
            if not os.path.exists(
                Pa_S
            ):  # Replace the MdlN of with the new one, so that we don't have to do it manually.
                sh.copy2(Pa_B, Pa_S)
                with open(Pa_S, 'r') as f1:
                    contents = f1.read()
                with open(Pa_S, 'w') as f2:
                    f2.write(contents.replace(MdlN_B, MdlN))
                if '.bat' not in Pa_B.lower():
                    os.startfile(
                        Pa_S
                    )  # Then we'll open it to make any other changes we want to make. Except if it's the BAT file
                vprint(f'🟢 - {Pa_S.split("/")[-1]} created successfully! (from {Pa_B})')
            else:
                print(
                    f'🟡 - {Pa_S.split("/")[-1]} already exists. If you want it to be replaced, you have to delete it manually before running this command.'
                )
        except Exception as e:
            print(f'🔴 - Error copying {Pa_B} to {Pa_S}: {e}')

    try:
        if not os.path.exists(
            Pa_PRJ
        ):  # For the PRJ file, there is no default text replacement to be performed, so we'll just copy.
            sh.copy2(Pa_PRJ_B, Pa_PRJ)
            os.startfile(Pa_PRJ)  # Then we'll open it to make any other changes we want to make.
            vprint(f'🟢 - {Pa_PRJ.split("/")[-1]} created successfully! (from {Pa_PRJ_B})')
        else:
            print(
                f'🟡 - {Pa_PRJ.split("/")[-1]} already exists. If you want it to be replaced, you have to delete it manually before running this command.'
            )
    except Exception as e:
        print(f'🔴 - Error copying {Pa_PRJ_B} to {Pa_PRJ}: {e}')
    vprint(Sign)


def S_from_B_undo(MdlN: str):
    """Will undo S_from_B by deletting S files"""
    vprint(Pre_Sign)
    d_Pa = get_MdlN_paths(MdlN)  # Get default directories
    MdlN_B, Pa_INI_B, Pa_INI, Pa_BAT_B, Pa_BAT, Pa_Smk, Pa_Smk_B, Pa_PRJ_B, Pa_PRJ = (
        d_Pa[k] for k in ['MdlN_B', 'INI_B', 'INI', 'BAT_B', 'BAT', 'Smk', 'Smk_B', 'PRJ_B', 'PRJ']
    )  # and pass them to objects that will be used in the function

    confirm = (
        input(f'Are you sure you want to delete the Cfg files (.smk, .ini, .bat, .prj) for {MdlN}? (y/n): ')
        .strip()
        .lower()
    )
    if confirm == 'y':
        for Pa_S in [Pa_Smk, Pa_BAT, Pa_INI, Pa_PRJ]:
            os.remove(Pa_S)  # Delete the S files
            vprint(f'🟢 - {Pa_S.split("/")[-1]} deleted successfully!')
    vprint(Sign)


def Up_log(MdlN: str, d_Up: dict, Pa_log=Pa_log):  # Pa_log=PJ(Pa_WS, 'Mng/log.csv')):
    """Update log.csv based on MdlN and key of `updates`."""
    Pa_lock = Pa_log + '.lock'  # Create a lock file to prevent concurrent access
    lock = FL(Pa_lock)

    with lock:  # Acquire the lock to prevent concurrent access
        DF = pd.read_csv(Pa_log, index_col=0)  # Assumes log.csv exists.

        for key, value in d_Up.items():  # Update the relevant cells
            DF.at[MdlN, key] = value

        while True:  # Wait for file to be closed if it's open
            try:
                DF.to_csv(Pa_log, date_format='%Y-%m-%d %H:%M')  # Save back to CSV
                break  # Break if successful
            except PermissionError:
                input('log.csv is open. Press Enter after closing the file...')  # Wait for user input


def _RunMng(args):
    """Helper function that runs a single model's snakemake workflow."""
    _, Se_Ln, cores_per_Sim, generate_dag = args
    Pa_Smk = PJ(Pa_WS, f'models/{Se_Ln["model alias"]}/code/snakemake/{Se_Ln["MdlN"]}.smk')
    Pa_Smk_log = PJ(
        Pa_WS,
        f'models/{Se_Ln["model alias"]}/code/snakemake/log/{Se_Ln["MdlN"]}_{DT.now().strftime("%Y%m%d_%H%M%S")}.log',
    )
    Pa_DAG = PJ(Pa_WS, f'models/{Se_Ln["model alias"]}/code/snakemake/DAG/DAG_{Se_Ln["MdlN"]}.png')
    vprint(f'{fg("cyan")}{PBN(Pa_Smk)}{attr("reset")}\n')

    try:
        if generate_dag:  # DAG parameter passed from RunMng
            sp.run(
                [
                    'snakemake',
                    '--dag',
                    '-s',
                    Pa_Smk,
                    '--cores',
                    str(cores_per_Sim),
                    '|',
                    'dot',
                    '-Tpng',
                    '-o',
                    f'{Pa_DAG}',
                ],
                shell=True,
                check=True,
            )
        with open(Pa_Smk_log, 'w', encoding='utf-8-sig') as f:
            sp.run(
                ['snakemake', '-p', '-s', Pa_Smk, '--cores', str(cores_per_Sim)],
                check=True,
                stdout=f,
                stderr=f,
            )
        return (Se_Ln['MdlN'], True)
    except sp.CalledProcessError as e:
        return (Se_Ln['MdlN'], False, str(e))


def RunMng(cores=None, DAG: bool = True, Cct_Sims=None):
    """
    Read the RunLog, and for each queued model, run the corresponding Snakemake file.

    Parameters:
        cores: Number of cores to allocate to each Snakemake process
        DAG: Whether to generate a DAG visualization
        Cct_Sims: Number of models to run simultaneously (defaults to number of available cores)
    """
    if cores is None:
        cores = max(
            cpu_count() - 2, 1
        )  # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    vprint(
        f'{Pre_Sign}RunMng initiated on {fg("cyan")}{str(DT.now()).split(".")[0]}{attr("reset")}. All Sims that are queued in the RunLog will be executed.\n'
    )

    vprint('Reading RunLog ...', end='')
    DF = read_RunLog()
    DF_q = DF.loc[
        ((DF['Start Status'] == 'Queued') & ((DF['End Status'].isna()) | (DF['End Status'] == 'Failed')))
    ]  # _q for queued. Only Run Queued runs that aren't running or have finished.
    vprint(' completed!\n')

    if not Cct_Sims:
        N_Sims = len(DF_q)
        Cct_Sims = max(
            min(N_Sims, cores), 1
        )  # Number of Sims to run simultaneously, limited by number of queued runs and available cores

    cores_per_Sim = cores // Cct_Sims  # Number of cores per Sim

    vprint(
        f'Found {fg("cyan")}{len(DF_q)} queued Sim(s){attr("reset")} in the RunLog. Will run {fg("cyan")}{Cct_Sims} Sim(s) simultaneously{attr("reset")}, using {bold}{cores_per_Sim} cores per Sim{bold_off}.\n'
    )

    if DF_q.empty:
        print('\n🟡🟡🟡 - No queued runs found in the RunLog.')
    else:
        # Prepare arguments for multiprocessing
        args = [(i, row, cores_per_Sim, DAG) for i, row in DF_q.iterrows()]

        # Run models in parallel
        with Pool(processes=Cct_Sims) as pool:
            results = pool.map(_RunMng, args)

        # Print results
        for result in results:
            if len(result) == 2:
                model_id, success = result
                if success:
                    vprint(f'🟢🟢 Model {model_id} completed successfully')
                else:
                    print(f'🔴🔴 Model {model_id} failed')
            else:
                model_id, success, error = result
                print(f'🔴🔴 Model {model_id} failed: {error}')

    vprint(Sign)


def reset_Sim(MdlN: str):
    """
    Resets the simulation by:
        1. Deleting all files in the MldN folder in the Sim folder.
        2. Clearing log.csv.
        3. Deletes Smk log files for MdlN.
        4. Deletes PoP folder for MdlN.
    """

    vprint(Pre_Sign)
    permission = (
        input(
            f'This will delete the Sim/{MdlN} folder and clear the corresponding line of the log.csv. Are you sure you want to proceed? (y/n): '
        )
        .strip()
        .lower()
    )

    if permission == 'y':
        d_Pa = get_MdlN_Pa(MdlN)  # Get default directories
        Pa_MdlN = d_Pa['Pa_MdlN']
        DF = pd.read_csv(Pa_log)  # Read the log file
        Pa_Smk_temp = d_Pa['Smk_temp']
        l_temp = [i for i in LD(Pa_Smk_temp) if MdlN in i]

        if (
            os.path.exists(Pa_MdlN) or (MdlN in DF['MdlN'].values) or l_temp or os.path.exists(d_Pa['PoP_Out_MdlN'])
        ):  # Check if the Sim folder exists or if the MdlN is in the log file
            i = 0

            try:
                if not os.path.exists(Pa_MdlN):
                    raise FileNotFoundError(f'{Pa_MdlN} does not exist.')
                sp.run(f'rmdir /S /Q "{Pa_MdlN}"', shell=True)  # Delete the entire Sim folder
                vprint('🟢 - Sim folder removed successfully.')
                i += 1
            except Exception as e:
                vprint(f'🔴 - failed to delete Sim folder: {e}')

            try:
                DF[DF['MdlN'] != MdlN].to_csv(Pa_log, index=False)  # Remove the log entry for this model
                vprint('🟢 - Log file updated successfully.')
                i += 1
            except Exception as e:
                vprint(f'🔴 - failed to update log file: {e}')

            try:
                for j in l_temp:
                    os.remove(PJ(Pa_Smk_temp, j))
                vprint('🟢 - Smk log files deleted successfully.')
                i += 1
            except Exception as e:
                vprint(f'🔴 - failed to remove Smk log files: {e}')

            try:
                if not os.path.exists(d_Pa['PoP_Out_MdlN']):
                    raise FileNotFoundError(f'{d_Pa["PoP_Out_MdlN"]} does not exist.')
                sp.run(f'rmdir /S /Q "{d_Pa["PoP_Out_MdlN"]}"', shell=True)  # Delete the entire Sim folder
                vprint('🟢 - PoP Out folder removed successfully.')
                i += 1
            except Exception as e:
                vprint(f'🔴 - failed to delete PoP Out folder: {e}')

            if i == 4:
                vprint('\n🟢🟢🟢 - ALL files were successfully removed.')
            else:
                vprint(f'🟡🟡🟡 - {i}/4 sub-processes finished successfully.')
        else:
            print(
                '🔴🔴🔴 - Items do not exist (Sim folder, log entry, Smk log files, PoP Out folder). No need to reset.'
            )
    else:
        print('🔴🔴🔴 - Reset cancelled by user (you).')
    vprint(Sign)


def rerun_Sim(MdlN: str, cores=None, DAG: bool = True):
    """
    Reruns the simulation by:
        1. Deleting all files in the MldN folder in the Sim folder.
        2. Clearing log.csv.
        3. Deletes Smk log files for MdlN.
        4. Deletes PoP folder for MdlN.
        5. Runs S_from_B to prepare the simulation files again.
    """

    if cores is None:
        cores = max(
            cpu_count() - 2, 1
        )  # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    reset_Sim(MdlN)

    DF = read_RunLog()

    if MdlN not in DF['MdlN'].values:
        print(f'🔴🔴🔴 - {MdlN} not found in the RunLog. Cannot rerun.')
        return
    else:
        Se_Ln = DF.loc[DF['MdlN'] == MdlN].squeeze()  # Get the row for the MdlN

        # Prepare arguments for multiprocessing
        args = [('_', Se_Ln, cores, DAG)]

        # Run models in parallel
        with Pool(processes=cores) as pool:
            results = pool.map(_RunMng, args)

        # Print results
        for result in results:
            if len(result) == 2:
                model_id, success = result
                if success:
                    vprint(f'🟢🟢 Model {model_id} completed successfully')
                else:
                    print(f'🔴🔴 Model {model_id} failed')
            else:
                model_id, success, error = result
                print(f'🔴🔴 Model {model_id} failed: {error}')

    vprint(Sign)


def get_elapsed_time_str(start_time: float) -> str:
    """Returns elapsed time as a formatted string.
    Format: 'd.hh:mm:ss' for days or 'hh:mm:ss' when less than a day"""
    elapsed = DT.now() - start_time
    s = int(elapsed.total_seconds())
    d, h, m, s = s // 86400, (s // 3600) % 24, (s // 60) % 60, s % 60

    if d:
        return f'{d}.{h:02}:{m:02}:{s:02}'
    return f'{h:02}:{m:02}:{s:02}'


# --------------------------------------------------------------------------------


# Edit common text files ---------------------------------------------------------
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
            vprint(f'🟢 - Added MOVER option to {PBN(Pa)}')
        except Exception as e:
            print(f'🔴 - Error adding MOVER option to {PBN(Pa)}: {e}')


# --------------------------------------------------------------------------------
