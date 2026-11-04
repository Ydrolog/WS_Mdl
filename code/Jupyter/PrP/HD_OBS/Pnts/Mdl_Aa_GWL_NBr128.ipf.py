"""Script to trim GWL_NBr8.ipf file to NBr128 MdlAa. Whole IPF file is heavy (can's select extent with imod.formats.ipf.read when reading it), so this is a workaround to make it faster to read."""

# %% 0. Imports
import pandas as pd
import WS_Mdl.core.df  # noqa: F401
from WS_Mdl.core.mdl import Mdl_N

# %% 1. Options
MdlN = 'NBr128'
M = Mdl_N(MdlN)  # load Mdl_N instance

Pa_In = M.Pa.In / 'OBS/HD_OBS_WEL/NBr8/GWL_NBr8.ipf'

DF = pd.read_csv(
    Pa_In,
    skiprows=14,
    names=[
        'X',
        'Y',
        'L',
        'Itsc_Thk',
        'Id',
        'code',
        'filterno',
        'surfacelevel',
        'filtertoplevel',
        'filterbottomlevel',
        'path',
    ],
)

# %%
DF_ = DF.loc[
    DF['X'].between(M.Xmin, M.Xmax, inclusive='both') & DF['Y'].between(M.Ymin, M.Ymax, inclusive='both')
]  # clip to Mdl area

DF_Out = DF_.copy()
_float_cols = DF_Out.select_dtypes(include=['float', 'Float64']).columns
if len(_float_cols) > 0:
    DF_Out.loc[:, _float_cols] = DF_Out.loc[:, _float_cols].round(3)


# %% Write file
Pa_Out = M.Pa.In / f'OBS/HD_OBS_WEL/{MdlN}/HD_OBS_Pnts_{MdlN}.ipf'
Pa_Out.parent.mkdir(parents=True, exist_ok=True)

with open(Pa_In, 'r') as f_in, open(Pa_Out, 'w', newline='') as f_out:
    # 1st line - number of entries, 2nd line - number of columns
    f_out.write(f'{DF_Out.shape[0]}\n')
    f_out.write(f'{DF_Out.shape[1]}\n')

    # Next 11 lines - headers
    f_out.writelines(f'{C}\n' for C in DF_Out.columns)

    # Number of columns (AGAIN!) and txt (don't get what the role of txt is, but it is in the original file)
    f_out.write(f'{DF_Out.shape[1]},txt\n')

    # Data
    DF_Out.to_csv(f_out, index=False, header=False, lineterminator='\n')

# %% Write metadata
Pa_Meta = M.Pa.In / f'OBS/HD_OBS_WEL/{MdlN}/HD_OBS_Pnts_{MdlN}_metadata.txt'

with open(Pa_Meta, 'w', newline='') as f_out:
    f_out.write(f'Created by: {__file__}\n')
    f_out.write(f'Source file: {Pa_In}\n')
    f_out.write(f'Trimmed to MdlAa: {MdlN}\n')
    f_out.write(f'Number of entries: {DF_Out.shape[0]}\n')
    f_out.write(f'Number of columns: {DF_Out.shape[1]}\n')
