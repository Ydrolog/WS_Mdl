from WS_Mdl.core import Sep
from WS_Mdl.core.path import Pa_WS
from WS_Mdl.io.ibridges import Upl, iB_session

print(Sep)

Mdl = 'NBr'

l_In = [
    f
    for f in (Pa_WS / 'models' / Mdl / 'In').glob('*')
    if (f.name != 'CAP') and (f.name != 'CHD') and (f.name.lower() != 'ss')
]  # CAP contains P and PET, which are composed of a very large number of ASCII files. That takes too long, so we'll upload their compressed versions (.tar.gz) instead.

for i in (Pa_WS / f'models/{Mdl}/In/CAP').glob('*'):  # Do CAP separetely to exclude 'P' and 'PET'
    if i.name != 'P' and i.name != 'PET':
        l_In.append(f'models/{Mdl}/In/CAP/{i.name}')

for i in (Pa_WS / f'models/{Mdl}/In/CAP/P').glob('*.tar.gz'):
    l_In.append(f'models/{Mdl}/In/CAP/P/{i.name}')

for i in (Pa_WS / f'models/{Mdl}/In/CAP/PET').glob('*.tar.gz'):
    l_In.append(f'models/{Mdl}/In/CAP/PET/{i.name}')

for i in (Pa_WS / f'models/{Mdl}/In/CHD').glob('*.tar.gz'):
    l_In.append(f'models/{Mdl}/In/CHD/{i.name}')

l_In.remove(Pa_WS / f'models/{Mdl}/In/DVC_check_Dup.ps1')  # Exclude this script from upload

l_Fo = [
    f'models/{Mdl}/code',
    f'models/{Mdl}/doc',
    f'models/{Mdl}/PrP',
] + l_In

print(
    f'Uploading:\n{"\n".join([f"{i:2}/{len(l_Fo)} - {j}" for i, j in enumerate(l_Fo, 1)])}\nfolder(s) to iBridges...\n'
)

S = iB_session()

for f in l_Fo:
    Upl(f, S)  # , overwrite=False)


print(Sep)
