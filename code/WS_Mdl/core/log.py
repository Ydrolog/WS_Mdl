# ---------- RunLog related functions ----------
import pandas as pd

from .path import Pa_log_Cfg
from .style import sprint, warn


def DF_match_MdlN(DF: pd.DataFrame, MdlN: str, Col_name='MdlN', case_insensitive=True):
    """Returns a boolean Series indicating which rows in the DataFrame match the given MdlN in the specified column."""
    if case_insensitive:
        return DF[Col_name].str.lower() == MdlN.lower()
    else:
        return DF[Col_name] == MdlN


def MdlN_Se_from_RunLog(
    MdlN,
):  # Can be made faster. May need to make excel export the RunLog as a csv, so that I can use pd.read_csv instead of pd.read_excel.
    """Returns RunLog line that corresponds to MdlN as a S."""

    DF = pd.read_csv(Pa_log_Cfg)
    Se_match = DF.loc[DF_match_MdlN(DF, MdlN)]  # Match MdlN, case insensitive.
    if Se_match.empty:
        print(f'🔴 - MdlN {MdlN} not found in RunLog.')
        sprint('Check the spelling and try again.', style=warn)
        raise ValueError()
    S = Se_match.squeeze()
    return S
