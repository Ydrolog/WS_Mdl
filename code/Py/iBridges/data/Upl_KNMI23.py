import WS_Mdl.utils as U

print(U.Sep)
Fo = 'data/KNMI23'

print(f'Uploading "{Fo}" folder to iBridges...\n')

S = U.iB_session()

U.iB_Upl(Fo, S)  # , overwrite=False)

print(U.Sep)
