from WS_Mdl.core import Sep
from WS_Mdl.io.ibridges import Upl, iB_session

print(Sep)

l_F = ['data/Dates.csv', 'data/.gitignore']

print(f'Uploading "{l_F}" to iBridges...\n')

S = iB_session()
S.info()

for F in l_F:
    Upl(F, S)  # , overwrite=False)

print(Sep)
