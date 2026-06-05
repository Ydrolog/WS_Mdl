from pathlib import Path
from WS_Mdl.core import Mdl_N, Sep, Pa_WS
from WS_Mdl.io.ibridges import Upl, iB_session

MdlN = 'NBr76'

print(Sep)

M = Mdl_N(MdlN)

S = iB_session()
S.info()

Upl(Path(*M.Pa.MdlN.parts[1:]), S, Pa_base='E:/')

print(f"Successfully uploaded {MdlN}!")

print(Sep)
