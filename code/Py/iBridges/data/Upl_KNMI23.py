import WS_Mdl.utils as U

print(U.pre_Sign)
Fo = 'data/KNMI23'

print(f'Uploading "{Fo}" folder to iBridges...\n')

S = U.iB_load_session()

U.iB_Upl(Fo, S)  # , overwrite=False)

print(U.post_Sign)
