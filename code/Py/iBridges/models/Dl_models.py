from WS_Mdl.core import Sep
from WS_Mdl.io.ibridges import Dl, iB_session

print(Sep)

l_F = ['models/NBr/code', 'models/NBr/doc', 'models/NBr/In']

print(f'Downloading "{l_F}" to from iBridges...\n')

S = iB_session()

S.info()

for F in l_F:
    Dl(F, S)  # , overwrite=False)

print(Sep)
