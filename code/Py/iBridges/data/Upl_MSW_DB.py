from WS_Mdl.core import Sep
from WS_Mdl.io.ibridges import Upl, iB_session

print(Sep)

l_F = ['data/MSW_DB.tar.gz']

print(f'Uploading "{l_F}" to iBridges...\n')

S = iB_session()
S.info()

for F in l_F:
    Upl(F, S, l_exceptions=[])  # , overwrite=False)

print(Sep)
