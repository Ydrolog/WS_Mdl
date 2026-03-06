# ---------- RunLog related functions ----------
from pathlib import Path

import pandas as pd

from .path import Pa_log_Cfg, Pa_log_Out, Pa_RunLog
from .style import sprint, warn


def DF_match_MdlN(DF: pd.DataFrame, MdlN: str, Col_name='MdlN', case_insensitive: bool = True):
    """Returns a boolean Series indicating which rows in the DataFrame match the given MdlN in the specified column."""
    if case_insensitive:
        return DF[Col_name].str.lower() == MdlN.lower()
    else:
        return DF[Col_name] == MdlN


def to_Se(MdlN: str):
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


def r_RunLog():
    return pd.read_excel(Pa_RunLog, sheet_name='RunLog').dropna(subset='runN')  # Read RunLog


def update_log(MdlN: str, d_Up: dict, Pa_log_Out=Pa_log_Out):  # Pa_log_Out=PJ(Pa_WS, 'Mng/log.csv')):
    """Update log_Out.csv based on MdlN and key of `updates`."""
    from filelock import FileLock as FL

    Pa_log_Out = Path(Pa_log_Out)
    Pa_lock = Pa_log_Out.with_name(f'{Pa_log_Out.name}.lock')  # Create a lock file to prevent concurrent access
    lock = FL(Pa_lock)

    with lock:  # Acquire the lock to prevent concurrent access
        DF = pd.read_csv(Pa_log_Out, index_col=0)  # Assumes log_Out.csv exists.

        for key, value in d_Up.items():  # Update the relevant cells
            DF.at[MdlN, key] = value

        while True:  # Wait for file to be closed if it's open
            try:
                DF.to_csv(Pa_log_Out, date_format='%Y-%m-%d %H:%M')  # Save back to CSV
                break  # Break if successful
            except PermissionError:
                input('log.csv is open. Press Enter after closing the file...')  # Wait for user input


def get_B(MdlN):
    """Returns the Baseline Sim for a given MdlN, based on the RunLog."""
    S = to_Se(MdlN)
    return S['B MdlN']
