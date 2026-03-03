# ---------- iMOD Point File Related Functions ----------

from pathlib import Path

import pandas as pd
from WS_Mdl.core.style import sprint


def r_as_DF(Pa_IPF: Path | str) -> pd.DataFrame:
    """
    Reads IPF file without temporal component - i.e. no linked TS text files. Returns a DF created from just the IPF file.
    """
    with open(Pa_IPF, 'r') as f:
        l_Ln = f.readlines()

    N_C = int(l_Ln[1].strip())  # Number of columns
    l_C_Nm = [l_Ln[i + 2].split('\n')[0] for i in range(N_C)]  # Extract column names
    DF_IPF = pd.read_csv(Pa_IPF, skiprows=2 + N_C + 1, names=l_C_Nm)

    sprint(
        f'🟢 - IPF file {Pa_IPF} read successfully. DataFrame created with {len(DF_IPF)} rows and {len(DF_IPF.columns)} columns.'
    )
    return DF_IPF
