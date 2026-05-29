from WS_Mdl.core import Pa_WS, Sep
from WS_Mdl.io.ibridges import Upl, iB_session

print(Sep)

Mdl = 'NBr'

l_In = [
    f.relative_to(Pa_WS)
    for f in (Pa_WS / 'models' / Mdl / 'In').glob('*')
    if (f.name != 'CAP') and (f.name != 'CHD') and (f.name.lower() != 'ss') and (f.name != 'OBS')
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

# l_In.remove(f'models/{Mdl}/In/DVC_check_Dup.ps1')  # Exclude this script from upload

l_Fo = [
    f'models/{Mdl}/code',
    f'models/{Mdl}/doc',
    f'models/{Mdl}/other',
    f'models/{Mdl}/PoP/In',
    f'models/{Mdl}/PoP/common/Ln',  # We want all folders from common, except DataPacks cause it's huge
    f'models/{Mdl}/PoP/common/Pgn',
    f'models/{Mdl}/PoP/common/Pt',
    f'models/{Mdl}/PoP/common/Rst',
    f'models/{Mdl}/PoP/common/symbology',
    f'models/{Mdl}/In/OBS/DRN',
    f'models/{Mdl}/In/OBS/HD',
    f'models/{Mdl}/In/OBS/RIV',
    f'models/{Mdl}/In/OBS/SFR',
    f'models/{Mdl}/In/OBS/HD_OBS_WEL/NBr5/obs',
    f'models/{Mdl}/In/OBS/HD_OBS_WEL/NBr5/ijkset_selectie.tar.gz',
    f'models/{Mdl}/In/OBS/HD_OBS_WEL/NBr7',
    f'models/{Mdl}/In/OBS/HD_OBS_WEL/NBr8',
] + l_In

print(
    f'Uploading:\n{"\n".join([f"{i:2}/{len(l_Fo)} - {j}" for i, j in enumerate(l_Fo, 1)])}\nfolder(s) to iBridges...\n'
)

S = iB_session()
S.info()

for f in l_Fo:
    print(f)
    Upl(f, S)  # , overwrite=False)


print(Sep)
