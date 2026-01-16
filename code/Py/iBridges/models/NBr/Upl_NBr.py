import os

import WS_Mdl.utils as U

print(U.pre_Sign)

Mdl = 'NBr'

l_In = [
    f'models/{Mdl}/In/{f}' for f in os.listdir(U.PJ(U.Pa_WS, f'models/{Mdl}/In')) if f != 'CAP' and f != 'CHD'
]  # CAP contains P and PET, which are composed of a very large number of ASCII files. That takes too long, so we'll upload their compressed versions (.tar.gz) instead.

for i in os.listdir(U.PJ(U.Pa_WS, f'models/{Mdl}/In/CAP')):
    if i != 'P' and i != 'PET':
        l_In.append(f'models/{Mdl}/In/CAP/{i}')

for i in os.listdir(U.PJ(U.Pa_WS, f'models/{Mdl}/In/CAP/P')):
    if i.endswith('.tar.gz'):
        l_In.append(f'models/{Mdl}/In/CAP/P/{i}')

for i in os.listdir(U.PJ(U.Pa_WS, f'models/{Mdl}/In/CAP/PET')):
    if i.endswith('.tar.gz'):
        l_In.append(f'models/{Mdl}/In/CAP/PET/{i}')

for i in os.listdir(U.PJ(U.Pa_WS, f'models/{Mdl}/In/CHD')):
    if i.endswith('.tar.gz'):
        l_In.append(f'models/{Mdl}/In/CHD/{i}')

l_In.remove(f'models/{Mdl}/In/DVC_check_Dup.ps1')  # Exclude this script from upload

l_Fo = [
    f'models/{Mdl}/code',
    f'models/{Mdl}/doc',
    f'models/{Mdl}/PrP',
] + l_In

print(
    f'Uploading:\n{"\n".join([f"{i:2}/{len(l_Fo)} - {j}" for i, j in enumerate(l_Fo, 1)])}\nfolder(s) to iBridges...\n'
)

S = U.iB_load_session()

for f in l_Fo:
    U.iB_Upl_Fo(f, S)  # , overwrite=False)

print(U.post_Sign)
