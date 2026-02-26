# ---------- RunLog related functions ----------
import pandas as pd

from .path import Pa_log_Cfg, Pa_log_Out
from .style import sprint, warn


def DF_match_MdlN(DF: pd.DataFrame, MdlN: str, Col_name='MdlN', case_insensitive: bool = True):
    """Returns a boolean Series indicating which rows in the DataFrame match the given MdlN in the specified column."""
    if case_insensitive:
        return DF[Col_name].str.lower() == MdlN.lower()
    else:
        return DF[Col_name] == MdlN


def MdlN_Se_from_RunLog(MdlN: str):
    """Returns RunLog line that corresponds to MdlN as a S."""

    DF = pd.read_csv(Pa_log_Cfg)
    Se_match = DF.loc[DF_match_MdlN(DF, MdlN)]  # Match MdlN, case insensitive.

    if Se_match.empty:
        print(f'🔴 - MdlN {MdlN} not found in RunLog.')
        sprint('Check the spelling and try again.', style=warn)
        raise ValueError()
    S = Se_match.squeeze()
    return S


def last_MdlN(status: str = 'completed'):
    # 666 to filter for Mdl in the future, and to be added as a method to MdlN class.
    """Returns the MdlN of the last completed or designed simulation, based on the RunLog."""
    if status == 'completed':
        DF = pd.read_csv(Pa_log_Out)
        DF.loc[:-2, 'Sim end DT'] = DF.loc[:-2, 'Sim end DT'].apply(pd.to_datetime, dayfirst=True)
        DF['Sim end DT'] = pd.to_datetime(DF['Sim end DT'], format='mixed', dayfirst=True)
        return DF.sort_values('Sim end DT', ascending=False).iloc[0]['MdlN']
    elif status == 'design':
        DF = pd.read_csv(Pa_log_Cfg)
        return DF['MdlN'].iloc[-1]
    else:
        print(f'🔴 - Invalid status: {status}. Use "completed" or "design".')
        raise ValueError()
