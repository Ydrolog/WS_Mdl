import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd
from WS_Mdl.core.path import Pa_WS
from WS_Mdl.core.style import sprint

# region MF6 -----------------------------------------------------------------------------------------------------------------------


def MF6_block_to_DF(
    Pa: Path | str,
    block: str,
    *,
    comment_chars: tuple[str, ...] = ('#', '!', '//'),
    has_header: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read the lines between ``BEGIN <block>`` and ``END <block>`` in a MODFLOW-6 file into a pandas DataFrame.

    A single comment line that **begins with ``#`` directly in front of the first data row** (i.e. before any non-comment, non-blank line is seen) is interpreted as the column names for the resulting DF, regardless of *has_header*.

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


# endregion MF6 --------------------------------------------------------------------------------------------------------------------


# region MSW -----------------------------------------------------------------------------------------------------------------------
def MSW_In_to_DF(Pa: Path | str) -> pd.DataFrame:
    """
    Converts the contents of a MSW input file into a pandas DataFrame.
    """
    Pa = Path(Pa)
    Fi = Pa.name

    d_headers = json.load(open(Path(Pa_WS) / 'code/WS_Mdl/Auxi/MSW_headers.json'))
    d_colspecs = json.load(open(Path(Pa_WS) / 'code/WS_Mdl/Auxi/MSW_colspecs.json'))

    try:
        if Fi == 'mete_grid.inp':
            DF = pd.read_csv(Pa, header=None, names=d_headers[Fi])
        elif Fi in d_colspecs:
            DF = pd.read_fwf(Pa, colspecs=d_colspecs[Fi], header=None)  # , l_headers=d_headers[Fi])
            DF.columns = d_headers[Fi][: DF.shape[1]]
        elif Fi == 'para_sim.inp':
            DF = pd.DataFrame({'Line': [ln.strip() for ln in open(Pa) if '=' in ln]})
            DF['parameter'] = DF['Line'].apply(lambda x: x.split('=')[0].strip())
            DF['value'] = DF['Line'].apply(lambda x: x.split('=')[1].split('!')[0].strip() if '=' in x else '')
            DF['comment'] = DF['Line'].apply(lambda x: x.split('!')[1].strip() if ('=' in x) and ('!' in x) else '')
            DF.drop(columns=['Line'], inplace=True)
        elif Fi == 'sel_key_svat_per.inp':
            DF = pd.DataFrame({'Line': [ln.strip() for ln in open(Pa)]})
            DF['parameter'] = DF['Line'].apply(lambda x: x.split()[0])
            DF['value'] = DF['Line'].apply(lambda x: x.split()[1])
            DF.drop(columns=['Line'], inplace=True)
        else:
            DF = pd.read_fwf(Pa, header=None)  # , l_headers=d_headers[i]
            DF.columns = d_headers[Fi][: DF.shape[1]]
        sprint(Fi, '🟢')
        return DF
    except Exception as e:
        sprint(Fi, '🔴', e)
        return


def mete_grid_to_DF(PRJ):
    DF_meteo = pd.read_csv(PRJ['extra']['paths'][2][0], names=['day', 'year', 'P', 'PET'])
    DF_meteo['DT'] = pd.to_datetime(
        DF_meteo['year'].astype(int).astype(str) + '-' + (DF_meteo['day'].astype(int) + 1).astype(str), format='%Y-%j'
    )
    return DF_meteo


# endregion ------------------------------------------------------------------------------------------------------------------------
