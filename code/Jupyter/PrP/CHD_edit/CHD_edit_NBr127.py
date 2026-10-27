"""
Increase CHD by 0.5 m to assess CHD impact area.
"""

# %% Imports
import imod
from WS_Mdl.core.mdl import Mdl_N

# %% Option
MdlN = 'NBr127'
M = Mdl_N(MdlN)
MdlN_CHD_B = 'NBr111'
MB = Mdl_N(MdlN_CHD_B)

# %% Iterate
for i in (MB.Pa.In / 'CHD' / MB.MdlN).glob('*199[12]*.idf'):
    A = imod.idf.open(i) + 0.5
    imod.idf.save(M.Pa.In / 'CHD' / M.MdlN / i.name.replace(MdlN_CHD_B, MdlN), A, pattern='{name}{extension}')

# %%
